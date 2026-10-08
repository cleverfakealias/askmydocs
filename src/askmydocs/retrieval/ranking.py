"""Pure ranking functions for keyword search and hybrid fusion. No model or database needed."""

import math
import re
from collections import Counter
from collections.abc import Sequence

from langchain_core.documents import Document

_TOKEN = re.compile(r"\w+")
_BM25_K1 = 1.5
_BM25_B = 0.75
_RRF_OFFSET = 60  # Standard constant from the reciprocal rank fusion paper.

type _FusionKey = tuple[object, object, str]


def tokenize(text: str) -> list[str]:
    """Return lowercase word tokens. Punctuation and whitespace are dropped."""
    return _TOKEN.findall(text.lower())


class BM25Index:
    """A BM25 keyword index over a fixed list of documents.

    Building the index tokenizes every document once. Build it again when the
    documents change.
    """

    def __init__(self, documents: Sequence[Document]) -> None:
        """Tokenize the documents and count term frequencies.

        Args:
            documents: The documents to search, in a fixed order.
        """
        self._documents = list(documents)
        self._term_counts = [Counter(tokenize(doc.page_content)) for doc in self._documents]
        self._lengths = [sum(counts.values()) for counts in self._term_counts]
        total = sum(self._lengths)
        self._average_length = total / len(self._lengths) if total else 1.0
        self._document_frequency: Counter[str] = Counter()
        for counts in self._term_counts:
            self._document_frequency.update(counts.keys())

    def __len__(self) -> int:
        """Return the number of indexed documents."""
        return len(self._documents)

    def rank(self, query: str, top_k: int) -> list[Document]:
        """Rank documents by BM25 score. Documents that share no term with the query are dropped.

        Args:
            query: The search text.
            top_k: The largest number of documents to return.

        Returns:
            Up to `top_k` documents, best match first.
        """
        query_terms = set(tokenize(query))
        if not query_terms or not self._documents:
            return []

        count = len(self._documents)
        scored: list[tuple[float, int]] = []
        for index, (term_counts, length) in enumerate(
            zip(self._term_counts, self._lengths, strict=True)
        ):
            norm = 1 - _BM25_B + _BM25_B * length / self._average_length
            score = 0.0
            for term in query_terms:
                frequency = term_counts.get(term, 0)
                if frequency == 0:
                    continue
                document_frequency = self._document_frequency[term]
                idf = math.log(1 + (count - document_frequency + 0.5) / (document_frequency + 0.5))
                score += idf * frequency * (_BM25_K1 + 1) / (frequency + _BM25_K1 * norm)
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._documents[index] for _, index in scored[:top_k]]


def bm25_rank(query: str, documents: Sequence[Document], top_k: int) -> list[Document]:
    """Rank documents by BM25 score in one call. Use `BM25Index` to search the same set often.

    Args:
        query: The search text.
        documents: The documents to search.
        top_k: The largest number of documents to return.

    Returns:
        Up to `top_k` documents, best match first.
    """
    return BM25Index(documents).rank(query, top_k)


def fuse_rankings(
    rankings: Sequence[tuple[Sequence[Document], float]], top_k: int
) -> list[Document]:
    """Combine ranked lists with weighted reciprocal rank fusion.

    A document scores higher when it ranks near the top of a list that has a large
    weight. Documents are matched by source, chunk index, and text.

    Args:
        rankings: Pairs of (ranked documents, weight).
        top_k: The largest number of documents to return.

    Returns:
        Up to `top_k` documents, best combined score first.
    """
    scores: dict[_FusionKey, float] = {}
    first_seen: dict[_FusionKey, Document] = {}

    for ranked, weight in rankings:
        for rank, doc in enumerate(ranked, start=1):
            key = (
                doc.metadata.get("source_id"),
                doc.metadata.get("chunk_index"),
                doc.page_content,
            )
            scores[key] = scores.get(key, 0.0) + weight / (_RRF_OFFSET + rank)
            first_seen.setdefault(key, doc)

    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [first_seen[key] for key in ordered[:top_k]]
