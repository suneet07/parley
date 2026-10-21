"""`parley` command line. Subcommands import lazily so `parley chat` never loads the audio stack."""

from __future__ import annotations

import argparse
import asyncio


def main() -> None:
    parser = argparse.ArgumentParser(prog="parley")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_chat = sub.add_parser("chat", help="talk to the agent brain in the terminal (text mode)")
    p_chat.add_argument("--business", help="path to business.yaml (default: BUSINESS_CONFIG)")
    p_chat.add_argument("--caller", help="caller phone number, for caller-ID lookup")

    p_serve = sub.add_parser("serve", help="run the API, browser demo and phone webhooks")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)

    sub.add_parser("ingest", help="index the business knowledge docs")

    p_eval = sub.add_parser("eval", help="run simulated-customer evals")
    p_eval.add_argument("--suite", default="smoke", help="smoke | full | <scenario id glob>")
    p_eval.add_argument("--concurrency", type=int, default=4)

    args = parser.parse_args()
    if args.cmd == "chat":
        from parley.core.brain.repl import run_repl

        asyncio.run(run_repl(business_path=args.business, caller_number=args.caller))
    elif args.cmd == "serve":
        from parley.api.server import run_server

        run_server(host=args.host, port=args.port)
    elif args.cmd == "ingest":
        from parley.core.knowledge.ingest import main as ingest_main

        asyncio.run(ingest_main())
    elif args.cmd == "eval":
        from evals.runner import main as eval_main

        asyncio.run(eval_main(suite=args.suite, concurrency=args.concurrency))


if __name__ == "__main__":
    main()
