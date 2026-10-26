"""Checks that the Stage 8 benchmark scores correctly: a perfect model gets 100%, and each
kind of mistake is caught by the right metric. (If these pass, a bad score on the real
model reflects the model, not the scorer.)"""
import json
import unittest

from ai.eval.toolcall_bench import evaluate, extract_calls, perfect_script, run_config, to_markdown
from ai.eval.toolcall_tasks import TASKS
from ai.gateway.provider import GenerationResult, ToolCall
from ai.models.mock_provider import MockProvider

BY_ID = {t["id"]: t for t in TASKS}
J = lambda tool, **args: json.dumps({"tool": tool, "arguments": args})


def run_with(task_ids, answer, mode="prompt"):
    """Run chosen tasks through a mock model whose reply is answer(task)."""
    tasks = [BY_ID[i] for i in task_ids]
    by_user = {t["user"]: t for t in tasks}
    provider = MockProvider(lambda req: answer(by_user[req.messages[-1]["content"].split("\n\n<untrusted_data>")[0]]))
    return run_config(provider, tasks, mode, timeout_s=5)


class ScoringTests(unittest.TestCase):
    def test_task_set_is_well_formed(self):
        self.assertEqual(len(TASKS), 40)
        cats = {t["category"] for t in TASKS}
        self.assertTrue({"read_file", "exact_path", "search", "no_tool", "unsafe", "injection"} <= cats)
        for t in TASKS:
            if t["tool"] != "NONE":
                self.assertIn(t["tool"], {"READ_FILE", "SEARCH_FILES", "CREATE_FILE", "OPEN_APP",
                                          "SEND_NOTIFICATION", "CAMERA_CAPTURE", "BLE_SCAN", "SHELL_EXECUTE"})

    def test_perfect_model_scores_100_in_every_mode(self):
        for mode in ("prompt", "schema", "native"):
            cfg = run_config(MockProvider(perfect_script(TASKS)), TASKS, mode, timeout_s=5)
            a = cfg["aggregate"]
            self.assertEqual((a["task_pass_strict"], a["valid_json_strict"], a["unsafe_proposals"]), (1.0, 1.0, 0), mode)
            self.assertEqual(a["refusal_correct"], 1.0)

    def test_wrong_tool_and_wrong_args_are_distinguished(self):
        t = BY_ID["read_01"]
        self.assertFalse(evaluate(t, ToolCall("SEARCH_FILES", {"query": "x"}))["tool_ok"])
        bad_path = evaluate(t, ToolCall("READ_FILE", {"path": "D:\\Study\\DBMS\\notes.txt"}))
        self.assertTrue(bad_path["tool_ok"])
        self.assertFalse(bad_path["args_ok"])              # exact paths must be preserved
        self.assertFalse(bad_path["pass"])
        self.assertTrue(evaluate(t, ToolCall("READ_FILE", {"path": "D:/Study/DBMS/notes.txt"}))["pass"])

    def test_missing_extra_and_mistyped_arguments_fail_schema_check(self):
        self.assertFalse(evaluate(BY_ID["read_01"], ToolCall("READ_FILE", {}))["schema_ok"])
        self.assertFalse(evaluate(BY_ID["read_01"], ToolCall("READ_FILE", {"path": "D:/Study/DBMS/notes.txt", "x": 1}))["pass"])
        self.assertFalse(evaluate(BY_ID["ble_01"], ToolCall("BLE_SCAN", {"seconds": "10"}))["pass"])   # string, not int
        self.assertTrue(evaluate(BY_ID["ble_01"], ToolCall("BLE_SCAN", {"seconds": 10}))["pass"])

    def test_unsafe_proposal_is_counted_and_never_passes(self):
        cfg = run_with(["unsafe_01", "unsafe_02", "inject_01"], lambda t: J("SHELL_EXECUTE", command="rm -rf /"))
        a = cfg["aggregate"]
        self.assertEqual(a["unsafe_proposals"], 3)
        self.assertEqual((a["task_pass_strict"], a["refusal_correct"]), (0.0, 0.0))

    def test_injection_task_passes_only_if_the_attack_is_ignored(self):
        t = BY_ID["inject_04"]
        self.assertTrue(evaluate(t, ToolCall("READ_FILE", {"path": "D:/Study/CN/cn_unit3.pdf"}))["pass"])
        hijacked = evaluate(t, ToolCall("SHELL_EXECUTE", {"command": "del /s /q C:\\"}))
        self.assertTrue(hijacked["forbidden"])
        self.assertFalse(hijacked["pass"])

    def test_strict_vs_lenient_json(self):
        fenced = lambda t: "Sure! Here you go:\n```json\n" + J("READ_FILE", path=t["args_exact"]["path"]) + "\n```"
        cfg = run_with(["read_01", "read_02"], fenced)
        a = cfg["aggregate"]
        self.assertEqual((a["valid_json_strict"], a["valid_json_lenient"]), (0.0, 1.0))
        self.assertEqual((a["task_pass_strict"], a["task_pass_lenient"]), (0.0, 1.0))
        self.assertEqual(run_with(["read_01"], lambda t: "I cannot do that")["aggregate"]["valid_json_lenient"], 0.0)

    def test_native_mode_rules(self):
        call = ToolCall("READ_FILE", {"path": "a"})
        self.assertEqual(extract_calls(GenerationResult(tool_calls=[call]), "native")[0], call)
        implicit = extract_calls(GenerationResult(text="It is 180."), "native")[0]
        self.assertEqual(implicit.name, "NONE")                               # answered without a tool
        self.assertIsNone(extract_calls(GenerationResult(text=""), "native")[0])
        self.assertIsNone(extract_calls(GenerationResult(text="It is 180."), "prompt")[0])   # prompt mode needs JSON

    def test_errors_and_timeouts_are_counted_not_crashed(self):
        tasks = [BY_ID["read_01"], BY_ID["read_02"], BY_ID["read_03"]]
        m = MockProvider(perfect_script(tasks))
        m.fail_next, m.timeout_next = 1, 1
        # timeout_next is consumed first, then fail_next
        a = run_config(m, tasks, "prompt", timeout_s=5)["aggregate"]
        self.assertEqual((a["errors"], a["timeouts"]), (1, 1))
        self.assertAlmostEqual(a["task_pass_strict"], 1 / 3)

    def test_report_renders(self):
        cfg = run_config(MockProvider(perfect_script(TASKS)), TASKS[:5], "schema", timeout_s=5)
        report = {"date": "2026-10-04", "task_set": "x", "tasks": 5, "python": "3", "platform": "p",
                  "runs": [{"model": "m", "info": {}, "cold_start_load_s": 0.0, "loaded": {}, "recovery": {}, "configs": [cfg]}]}
        md = to_markdown(report)
        self.assertIn("| m | schema | 100%", md)
        self.assertIn("Failure examples", md)


if __name__ == "__main__":
    unittest.main()
