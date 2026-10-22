"""Try the whole AI side end to end in a terminal.

    py -m ai.pipeline.demo --model qwen2.5:3b            # real local model through Ollama
    py -m ai.pipeline.demo --mock                        # scripted model, no Ollama needed
    py -m ai.pipeline.demo --model qwen2.5:3b --embedder fastembed --db demo_memory.db

Type a message and press Enter. Commands:
    :remember <text>            save a note you typed yourself (trusted)
    :import <path> | <text>     pretend a file was imported (untrusted, quarantined)
    :promote <memory_id>        trust a quarantined memory (use :promote! for injection-flagged ones)
    :revoke <path>              hide everything that came from that file
    :quit
The "policy" shown below is a tiny STAND-IN so the demo is readable. The real policy engine
belongs to Member 4; nothing here is authorization.
"""
from __future__ import annotations

import argparse
import itertools
import json

from ..api import AIModule
from ..contracts import INGEST_FILE_IMPORT, INGEST_USER_TYPED, Event
from ..models.embedder import HashingEmbedder
from ..models.mock_provider import MockProvider
from ..models.ollama_provider import OllamaProvider
from .agent import AgentPipeline, AgentTurn

READ_ONLY = {"READ_FILE", "SEARCH_FILES"}
_ids = itertools.count(1)


def stand_in_policy(turn: AgentTurn) -> str:
    call = turn.proposal
    if call.name == "SHELL_EXECUTE":
        return "DENY (shell is never allowed in this demo)"
    if turn.tainted_by and call.name not in READ_ONLY:
        return f"DENY (proposed after reading untrusted memory {turn.tainted_by})"
    return "ALLOW (read-only)" if call.name in READ_ONLY else "ASK THE USER (changes something)"


def event(text: str, ingest: str, **payload) -> Event:
    n = next(_ids)
    return Event(event_id=f"demo{n}", entity_id=f"n{n}", source_device_id="demo-pc", device_sequence=n,
                 event_type="file" if ingest == INGEST_FILE_IMPORT else "note", ingest_path=ingest,
                 payload={"text": text, **payload})


def show(turn: AgentTurn, ai: AIModule) -> None:
    r = turn.route
    print(f"  route    : {r.route.value} (stage {r.stage}, {r.reason})")
    for mid in turn.evidence_ids:
        row = ai.store.get(mid)
        print(f"  evidence : {mid} tier={row['trust_tier']} quarantined={bool(row['quarantined'])}")
    if turn.kind == "answer":
        print(f"  ANSWER   : {turn.text}")
    elif turn.kind == "proposal":
        print(f"  PROPOSAL : {turn.proposal.name} {json.dumps(turn.proposal.arguments)}")
        print(f"  policy   : {stand_in_policy(turn)}")
    else:
        print(f"  ERROR    : {turn.text} {turn.problems}")
    print(f"  tainted_by={turn.tainted_by} degraded={turn.degraded} model_calls={turn.model_calls} "
          f"time={turn.timings_s.get('total', 0):.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--db", default=":memory:")
    ap.add_argument("--embedder", choices=["hashing", "fastembed"], default="hashing")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    embedder, reranker, mode = HashingEmbedder(), None, "hybrid"
    if args.embedder == "fastembed":
        from ..models.fastembed_adapters import FastEmbedEmbedder, FastEmbedReranker
        embedder, reranker, mode = FastEmbedEmbedder(), FastEmbedReranker(), "hybrid_rerank"
    provider = (MockProvider(lambda req: '{"tool": "NONE", "arguments": {"reason": "answer"}}'
                             if req.json_schema is not None else "(mock answer)")
                if args.mock else OllamaProvider(args.model, args.host))
    health = provider.health()
    if not health.ok:
        raise SystemExit(f"Model not ready: {health.detail}")

    with AIModule(args.db, embedder=embedder, reranker=reranker) as ai:
        agent = AgentPipeline(ai, provider, retrieval_mode=mode)
        print(__doc__.split("Type a message")[0].strip().splitlines()[0])
        print("Type a message, or :remember / :import / :promote / :revoke / :quit\n")
        while True:
            try:
                line = input("you> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line == ":quit":
                break
            try:
                if line.startswith(":remember "):
                    r = ai.admit(event(line[10:], INGEST_USER_TYPED))
                    print(f"  {r.decision.value}: {r.reason} {r.memory_ids}")
                elif line.startswith(":import "):
                    path, _, text = line[8:].partition("|")
                    r = ai.admit(event(text.strip(), INGEST_FILE_IMPORT, path=path.strip()))
                    print(f"  {r.decision.value}: {r.reason} {r.memory_ids}")
                elif line.startswith(":promote"):
                    ack = line.startswith(":promote!")
                    ai.promote_memory(line.split()[1], confirmed=True, acknowledge_injection_flag=ack)
                    print("  promoted")
                elif line.startswith(":revoke "):
                    print(f"  revoked {ai.revoke_artifact(line[8:].strip())} memories")
                else:
                    show(agent.handle(line), ai)
            except (ValueError, KeyError, IndexError) as e:
                print(f"  not done: {e}")


if __name__ == "__main__":
    main()
