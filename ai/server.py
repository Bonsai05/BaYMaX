"""Standalone dev server for the AI module, so Members 1 and 2 can build against real endpoints
before the Core exists. The finished Core mounts `ai.service.build_router` itself instead.

    pip install fastapi uvicorn
    py -m ai.server                                  # real model + real embeddings, http://127.0.0.1:8765
    py -m ai.server --embedder hashing               # skip the embedding download
    py -m ai.server --host 0.0.0.0 --key mysecret    # reachable from a phone on the LAN; requests need the key

Open http://127.0.0.1:8765/docs for an interactive page that lists every endpoint.
With --key, send the header  X-Baymax-Key: mysecret  on every request. This is only a development
convenience; real authentication and device pairing belong to the Core (Member 1).
"""
from __future__ import annotations

import argparse
import hmac
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException

from .assistant import Assistant, AssistantConfig, build_assistant
from .service import build_router


def create_app(assistant: Assistant, key: Optional[str] = None) -> FastAPI:
    def check_key(x_baymax_key: Optional[str] = Header(default=None)) -> None:
        if key is not None and not hmac.compare_digest(x_baymax_key or "", key):
            raise HTTPException(status_code=401, detail="bad_or_missing_key")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        assistant.close()                      # release the database on shutdown

    app = FastAPI(title="BAYMAX AI module (dev server)", lifespan=lifespan)
    app.include_router(build_router(assistant), dependencies=[Depends(check_key)])
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--db", default="vault/memory.db")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--embedder", choices=["fastembed", "hashing"], default="fastembed")
    ap.add_argument("--key", default=None, help="require this value in the X-Baymax-Key header")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    if args.host not in ("127.0.0.1", "localhost") and not args.key:
        raise SystemExit("Refusing to listen on a network address without --key. Add --key <secret>.")
    import os
    if args.db != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(args.db)), exist_ok=True)
    assistant = build_assistant(AssistantConfig(db_path=args.db, model=args.model, embedder=args.embedder))
    if not args.no_warmup:
        print("Loading the model (first call can take 10 to 30 seconds)...")
        print(f"Warm-up took {assistant.warm_up():.1f}s")
    import uvicorn
    uvicorn.run(create_app(assistant, args.key), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
