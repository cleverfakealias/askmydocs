"""Command-line interface: `palimpsest ingest`, `ask`, `status`, and `ui`."""

import argparse
import logging
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from palimpsest.config import Settings
from palimpsest.engine import RAGEngine
from palimpsest.models import chunk_label

APP_PATH = Path(__file__).with_name("app.py")


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the `palimpsest` command."""
    parser = argparse.ArgumentParser(prog="palimpsest", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logs")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="index one or more documents")
    ingest.add_argument("files", nargs="+", type=Path, help="PDF, TXT, MD, or DOCX files")

    ask = commands.add_parser("ask", help="ask a question about the indexed documents")
    ask.add_argument("question", help="the question, in quotes")

    commands.add_parser("status", help="show indexed files and chunk counts")
    commands.add_parser("ui", help="open the Streamlit web interface on this machine")
    return parser


def ui_command() -> list[str]:
    """Return the command that starts the web UI.

    The server listens on localhost only, because the UI has no login. Streamlit's
    usage statistics are turned off.
    """
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.address=localhost",
        "--browser.gatherUsageStats=false",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    if args.command == "ui":
        return _run_ui()
    if args.command == "ask" and not args.question.strip():
        parser.error("the question is empty")  # Exits before any model loads.

    engine = RAGEngine.create(Settings())
    if args.command == "ingest":
        return _ingest(engine, args.files)
    if args.command == "ask":
        return _ask(engine, args.question)
    return _status(engine)


def _ingest(engine: RAGEngine, files: list[Path]) -> int:
    report = engine.ingest_files(files)
    for item in report.files:
        if item.ok:
            print(f"ok      {item.path} ({chunk_label(item.chunks)})")
        else:
            print(f"failed  {item.path}: {item.error}")
    print(f"\n{chunk_label(report.total_chunks)} stored.")
    return 1 if report.failed else 0


def _ask(engine: RAGEngine, question: str) -> int:
    answer = engine.ask(question, engine.new_conversation())
    print(answer.text)
    if answer.sources:
        print("\nSources:")
        for number, source in enumerate(answer.sources, start=1):
            print(f"  [{number}] {source.label}")
    return 0


def _status(engine: RAGEngine) -> int:
    sources = engine.indexed_sources()
    print(f"Embedding model: {engine.settings.resolved_embedding_model}")
    print(f"Chat model:      {engine.settings.resolved_llm_model}")
    print(f"Retrieval:       {engine.settings.retrieval_method}")
    print(f"Indexed files:   {len(sources)} ({chunk_label(engine.chunk_count())})")
    for source in sources:
        print(f"  {source.name}: {chunk_label(source.chunks)}  [{source.source_id}]")
    return 0


def _run_ui() -> int:
    try:
        # A fixed argument list with no shell and no user input.
        return subprocess.call(ui_command())  # noqa: S603
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
