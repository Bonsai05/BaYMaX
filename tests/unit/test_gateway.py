"""Stage 8 tests: provider interface, tool-call helpers, Ollama adapter (against a fake
local server, because the real model is run on the user's PC), fallback and metrics.

    python -m unittest discover -s tests -t . -v
"""
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ai.gateway.provider import (
    FallbackProvider, GenerationRequest, MeteredProvider, ToolCall,
)
from ai.gateway.toolcall import (
    DEFAULT_TOOLS, NO_TOOL, action_schema, build_system_prompt, parse_action, tools_for_ollama,
    validate_call,
)
from ai.models.mock_provider import MockProvider
from ai.models.ollama_provider import OllamaProvider

SEEN: list[dict] = []     # request bodies received by the fake server
MODE = {"value": "normal"}


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/tags":
            self._json(200, {"models": [{"name": "qwen2.5:3b"}, {"name": "tiny:latest"}]})
        elif self.path == "/api/ps":
            self._json(200, {"models": [{"name": "qwen2.5:3b", "size": 2_500_000_000, "size_vram": 2_400_000_000}]})
        else:
            self._json(404, {"error": "nope"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(body)
        mode = MODE["value"]
        if mode == "http_error":
            return self._json(404, {"error": "model 'x' not found"})
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        if mode == "slow":
            time.sleep(1.0)
        lines = []
        if mode == "tool":
            lines.append({"message": {"content": "", "tool_calls": [
                {"function": {"name": "READ_FILE", "arguments": {"path": "D:/a.txt"}}}]}, "done": False})
        elif mode == "midstream_error":
            lines.append({"message": {"content": "par"}, "done": False})
            lines.append({"error": "out of memory"})
        else:
            lines += [{"message": {"content": c}, "done": False} for c in ('{"tool": ', '"NONE", ', '"arguments": {}}')]
        if mode != "midstream_error":
            lines.append({"message": {"content": ""}, "done": True, "done_reason": "stop",
                          "prompt_eval_count": 40, "eval_count": 12, "load_duration": 500_000_000})
        for obj in lines:
            self.wfile.write((json.dumps(obj) + "\n").encode())
            self.wfile.flush()


class OllamaAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        cls.host = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        SEEN.clear()
        MODE["value"] = "normal"
        self.p = OllamaProvider("qwen2.5:3b", self.host)
        self.req = GenerationRequest([{"role": "user", "content": "hi"}], timeout_s=5)

    def test_stream_and_timings(self):
        chunks = list(self.p.stream(self.req))
        self.assertEqual("".join(c.delta for c in chunks), '{"tool": "NONE", "arguments": {}}')
        r = chunks[-1].result
        self.assertTrue(r.ok)
        self.assertEqual((r.timings.prompt_tokens, r.timings.output_tokens), (40, 12))
        self.assertAlmostEqual(r.timings.load_s, 0.5)
        self.assertIsNotNone(r.timings.ttft_s)
        self.assertLessEqual(r.timings.ttft_s, r.timings.total_s)

    def test_generate_matches_stream_text(self):
        self.assertEqual(self.p.generate(self.req).text, '{"tool": "NONE", "arguments": {}}')

    def test_request_options_schema_and_tools_are_forwarded(self):
        schema = action_schema(DEFAULT_TOOLS)
        req = GenerationRequest([{"role": "user", "content": "x"}], json_schema=schema,
                                tools=tools_for_ollama(DEFAULT_TOOLS), temperature=0.0, max_tokens=64)
        self.p.generate(req)
        body = SEEN[-1]
        self.assertEqual(body["format"], schema)
        self.assertEqual(body["options"]["num_predict"], 64)
        self.assertEqual(body["options"]["temperature"], 0.0)
        self.assertEqual(body["tools"][0]["function"]["name"], "READ_FILE")
        self.assertTrue(body["stream"])

    def test_native_tool_call_is_parsed(self):
        MODE["value"] = "tool"
        r = self.p.generate(self.req)
        self.assertEqual(r.tool_calls, [ToolCall("READ_FILE", {"path": "D:/a.txt"})])
        self.assertIsNotNone(r.timings.ttft_s)

    def test_slow_server_times_out_without_raising(self):
        MODE["value"] = "slow"
        r = self.p.generate(GenerationRequest([{"role": "user", "content": "x"}], timeout_s=0.3))
        self.assertEqual(r.finish_reason, "timeout")
        self.assertFalse(r.ok)

    def test_errors_never_raise(self):
        MODE["value"] = "http_error"
        r = self.p.generate(self.req)
        self.assertEqual(r.finish_reason, "error")
        self.assertIn("404", r.error)
        MODE["value"] = "midstream_error"
        r = self.p.generate(self.req)
        self.assertIn("out of memory", r.error)
        dead = OllamaProvider("m", "http://127.0.0.1:1")
        r = dead.generate(self.req)
        self.assertEqual(r.finish_reason, "error")
        self.assertIn("unreachable", r.error)

    def test_health(self):
        h = self.p.health()
        self.assertTrue(h.ok)
        self.assertEqual(h.loaded_models[0]["size_vram"], 2_400_000_000)
        self.assertTrue(OllamaProvider("tiny", self.host).health().ok)         # matches tiny:latest
        missing = OllamaProvider("llama3.2:3b", self.host).health()
        self.assertFalse(missing.ok)
        self.assertIn("ollama pull llama3.2:3b", missing.detail)
        self.assertFalse(OllamaProvider("m", "http://127.0.0.1:1").health().ok)

    def test_fallback_to_mock_when_ollama_is_down(self):
        dead = OllamaProvider("m", "http://127.0.0.1:1")
        backup = MockProvider(['{"tool": "NONE", "arguments": {}}'], name="backup")
        fb = FallbackProvider([dead, backup])
        r = fb.generate(self.req)
        self.assertTrue(r.ok)
        self.assertEqual((r.provider, r.fallback_from), ("backup", ["ollama"]))
        chunks = list(fb.stream(self.req))
        self.assertEqual(chunks[-1].result.fallback_from, ["ollama"])
        self.assertTrue(fb.health().ok)


class MockAndMetricsTests(unittest.TestCase):
    req = GenerationRequest([{"role": "user", "content": "x"}])

    def test_failure_injection_and_all_providers_failed(self):
        m = MockProvider(["ok"])
        m.fail_next = 1
        self.assertEqual(m.generate(self.req).finish_reason, "error")
        self.assertTrue(m.generate(self.req).ok)
        m.timeout_next = 1
        self.assertEqual(m.generate(self.req).finish_reason, "timeout")
        a, b = MockProvider(["x"], name="a"), MockProvider(["y"], name="b")
        a.fail_next = b.fail_next = 1
        r = FallbackProvider([a, b]).generate(self.req)
        self.assertFalse(r.ok)
        self.assertEqual(r.fallback_from, ["a"])

    def test_metrics_summary(self):
        a = MockProvider(["x"])
        metered = MeteredProvider(a)
        metered.generate(self.req)
        a.fail_next = 1
        metered.generate(self.req)
        s = metered.metrics.summary()
        self.assertEqual((s["calls"], s["errors"], s["timeouts"]), (2, 1, 0))
        self.assertEqual(MeteredProvider(MockProvider(["x"])).metrics.summary(), {"calls": 0})


class ToolCallTests(unittest.TestCase):
    def test_strict_and_lenient_parsing(self):
        good = '{"tool": "READ_FILE", "arguments": {"path": "a.txt"}}'
        self.assertEqual(parse_action(good)[0], ToolCall("READ_FILE", {"path": "a.txt"}))
        fenced = "Sure!\n```json\n" + good + "\n```\nHope that helps."
        self.assertIsNone(parse_action(fenced)[0])                       # strict rejects
        self.assertEqual(parse_action(fenced, lenient=True)[0].name, "READ_FILE")
        self.assertEqual(parse_action("not json")[1][:12], "invalid_json")
        self.assertEqual(parse_action('{"arguments": {}}')[1], "missing_tool_field")
        self.assertEqual(parse_action('{"tool": "X", "arguments": []}')[1], "arguments_not_object")
        braces = 'text {"tool": "OPEN_APP", "arguments": {"app": "a}b"}} more'
        self.assertEqual(parse_action(braces, lenient=True)[0].arguments, {"app": "a}b"})

    def test_validation(self):
        v = lambda name, **a: validate_call(ToolCall(name, a), DEFAULT_TOOLS)
        self.assertEqual(v("READ_FILE", path="a.txt"), [])
        self.assertEqual(v(NO_TOOL, reason="unsafe"), [])
        self.assertEqual(v("FORMAT_DISK"), ["unknown_tool:FORMAT_DISK"])
        self.assertEqual(v("READ_FILE"), ["missing_argument:path"])
        self.assertEqual(v("READ_FILE", path="a", mode="w"), ["unexpected_argument:mode"])
        self.assertEqual(v("READ_FILE", path=5), ["wrong_type:path"])
        self.assertEqual(v("BLE_SCAN", seconds=True), ["wrong_type:seconds"])   # bool is not an integer
        self.assertEqual(v("BLE_SCAN", seconds=5), [])

    def test_prompt_and_schema_mention_every_tool(self):
        prompt, schema = build_system_prompt(DEFAULT_TOOLS), action_schema(DEFAULT_TOOLS)
        for s in DEFAULT_TOOLS:
            self.assertIn(s.name, prompt)
            self.assertIn(s.name, schema["properties"]["tool"]["enum"])
        self.assertIn(NO_TOOL, schema["properties"]["tool"]["enum"])


if __name__ == "__main__":
    unittest.main()
