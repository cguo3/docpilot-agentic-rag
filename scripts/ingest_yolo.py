"""Ingest a HuggingFace corpus subset into a FAISS vector store.

Each (dataset, subset, split, embed_model) combination gets its own index
directory, so multiple models can be evaluated side-by-side.

Usage examples:
    python scripts/ingest_yolo.py
    python scripts/ingest_yolo.py --embed-model BAAI/bge-base-en-v1.5
    python scripts/ingest_yolo.py --embed-model text-embedding-3-small --embedder-type openai
    python scripts/ingest_yolo.py --force          # re-ingest even if cached
    python scripts/ingest_yolo.py --list-indexes   # show all persisted indexes
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

from tqdm import tqdm

from docpilot.cache import DatasetCache
from docpilot.embedder.base import EmbedderBase
from docpilot.indexing import (
    Cleaner,
    HuggingFaceLoader,
    MetadataExtractor,
    PassThroughChunker,
)
from docpilot.vector_store import FAISSStore

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent / "data"

# ── defaults ────────────────────────────────────────────────────────────────

DATASET_NAME   = "freshstack/corpus-oct-2024"
DATASET_SUBSET = "yolo"
SPLIT          = "train"
EMBED_MODEL    = "BAAI/bge-small-en-v1.5"


# ── progress helpers ─────────────────────────────────────────────────────────

@contextmanager
def _stage(label: str, total_stages: int, step: int):
    """Print a timed stage header that plays well with nested tqdm bars."""
    tqdm.write(f"\n[{step}/{total_stages}] {label}")
    t0 = time.monotonic()
    yield
    tqdm.write(f"    ✓ done in {time.monotonic() - t0:.1f}s")


# ── helpers ──────────────────────────────────────────────────────────────────

def _persist_dir(root: Path, subset: str, embed_model: str) -> str:
    return str(root / subset / embed_model.replace("/", "_"))


def _build_embedder(embed_model: str, embedder_type: str | None) -> EmbedderBase:
    use_openai = embedder_type == "openai" or (
        embedder_type is None and embed_model.startswith("text-embedding")
    )
    if use_openai:
        from docpilot.embedder.openai_embedding import OpenAIEmbedder
        return OpenAIEmbedder(model=embed_model)
    from docpilot.embedder.bge_embedding import BGEEmbedder
    return BGEEmbedder(model_name=embed_model)


def _list_indexes(root: Path) -> None:
    indexes = sorted(root.glob("*/*/index_meta.json"))
    if not indexes:
        tqdm.write(f"No persisted indexes found under {root}")
        return
    tqdm.write("Persisted indexes:")
    for meta_path in indexes:
        meta = json.loads(meta_path.read_text())
        tqdm.write(
            f"  {meta_path.parent.relative_to(root)}"
            f"  |  model={meta.get('embed_model', '?')}"
            f"  dims={meta.get('dimensions', '?')}"
            f"  docs={meta.get('doc_count', '?')}"
        )


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a HuggingFace corpus into FAISS (one index per model)."
    )
    parser.add_argument("--dataset",       default=DATASET_NAME,   help="HuggingFace dataset id")
    parser.add_argument("--subset",        default=DATASET_SUBSET, help="Dataset config / subset name")
    parser.add_argument("--split",         default=SPLIT,          help="Dataset split (default: train)")
    parser.add_argument("--embed-model",   default=EMBED_MODEL,    help="Embedding model identifier")
    parser.add_argument(
        "--embedder-type", choices=["bge", "openai"], default=None,
        help="Embedder backend. Auto-detected from model name if omitted.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-ingest even if this (dataset, model) combination is already cached.",
    )
    parser.add_argument(
        "--data-root", default=None,
        help=f"Root directory for indexes and cache (default: {_ROOT})",
    )
    parser.add_argument(
        "--list-indexes", action="store_true",
        help="Print all persisted indexes and exit.",
    )
    return parser.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────

async def main(args: Namespace | None = None) -> None:
    if args is None:
        args = _parse_args()

    root = Path(args.data_root) if args.data_root else _ROOT

    if args.list_indexes:
        _list_indexes(root)
        return

    persist_dir = _persist_dir(root, args.subset, args.embed_model)
    cache_dir   = str(root / "cache")
    cache       = DatasetCache(cache_dir=cache_dir)

    if not args.force and cache.is_cached(
        args.dataset, split=args.split, name=args.subset, embed_model=args.embed_model
    ):
        _log.warning(
            "Dataset '%s' (subset=%s, split=%s) was already ingested with model '%s'. "
            "Skipping. Use --force to re-ingest, or delete %s to clear the cache.",
            args.dataset, args.subset, args.split, args.embed_model, cache_dir,
        )
        tqdm.write(
            f"⚠  Already ingested — skipping.\n"
            f"   Use --force to re-embed, or delete {cache_dir} to clear the cache."
        )
        return

    tqdm.write(
        f"\nIngestion pipeline\n"
        f"  dataset : {args.dataset}  subset={args.subset}  split={args.split}\n"
        f"  model   : {args.embed_model}\n"
        f"  output  : {persist_dir}\n"
    )

    total_start = time.monotonic()
    loader    = HuggingFaceLoader(
        dataset_name=args.dataset, name=args.subset, split=args.split,
        text_field="text", id_field="_id", metadata_field="metadata",
    )
    cleaner   = Cleaner()
    extractor = MetadataExtractor()
    chunker   = PassThroughChunker()
    embedder  = _build_embedder(args.embed_model, args.embedder_type)
    store     = FAISSStore(
        dimensions=embedder.dimensions,
        persist_directory=persist_dir,
        embed_model=args.embed_model,
    )

    # ── Stage 1: Load ────────────────────────────────────────────────────────
    with _stage(f"Loading {args.dataset} ({args.subset}/{args.split})", 4, 1):
        documents = await loader.load()
        tqdm.write(f"    {len(documents):,} documents loaded")

    # ── Stage 2: Preprocess ──────────────────────────────────────────────────
    with _stage("Preprocessing (clean → extract metadata → chunk)", 4, 2):
        documents = await cleaner.clean(documents)
        documents = await extractor.extract(documents)
        chunks    = await chunker.chunk(documents)
        tqdm.write(f"    {len(chunks):,} chunks ready")

    # ── Stage 3: Embed ───────────────────────────────────────────────────────
    with _stage(f"Embedding with {args.embed_model}", 4, 3):
        embeddings = await embedder.embed_documents(chunks)
        tqdm.write(f"    {len(embeddings):,} vectors (dim={embedder.dimensions})")

    # ── Stage 4: Index ───────────────────────────────────────────────────────
    with _stage(f"Indexing into FAISS → {persist_dir}", 4, 4):
        ids = await store.ingest(chunks, embeddings)
        tqdm.write(f"    {len(ids):,} documents indexed")

    cache.mark_cached(
        args.dataset, split=args.split, name=args.subset,
        embed_model=args.embed_model, doc_count=len(ids),
    )

    total = time.monotonic() - total_start
    mins, secs = divmod(int(total), 60)
    tqdm.write(f"\n✓ Done — {len(ids):,} documents in {mins}m {secs}s\n")


if __name__ == "__main__":
    asyncio.run(main())
