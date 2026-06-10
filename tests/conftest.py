"""Shared fixtures for the test suite."""
from __future__ import annotations

import numpy as np
import pytest

from docpilot.core.schema import Document


def make_doc(id: str = "doc-1", text: str = "hello world", **meta) -> Document:
    return Document(id=id, text=text, metadata=meta)


def unit_vector(dim: int, seed: int = 0) -> list[float]:
    """Return a reproducible unit vector of given dimension."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(id="a", text="YOLO object detection", metadata={"topic": "cv"}),
        Document(id="b", text="Transformer architecture", metadata={"topic": "nlp"}),
        Document(id="c", text="RAG retrieval pipeline", metadata={"topic": "nlp"}),
    ]


@pytest.fixture
def sample_embeddings(sample_docs) -> list[list[float]]:
    """One unit vector per doc, reproducible by index."""
    return [unit_vector(dim=4, seed=i) for i in range(len(sample_docs))]
