"""
Ghosn Al-Qana AI — Knowledge Base Organizer
=============================================
Task: "Organize the date palm thinning knowledge base."
      (Issue #7 — Implement Memory and Knowledge Base)

This module reads Knowledge/theinning_guidelines.txt and splits it into
clean, numbered "chunks" — one chunk per section (e.g. "1. PURPOSE OF FRUIT
THINNING", "4. COLOR SENSOR", ...). This is the first, retrieval-ready
building block for the RAG pipeline (step 4 in our plan). It doesn't do
any embedding/vector search yet — it just organizes the raw text into
clean structured pieces that a retriever (keyword search now, FAISS/Chroma
later) can operate on.

Usage:
    from memory.knowledge_base import load_knowledge_base

    chunks = load_knowledge_base()
    for c in chunks:
        print(c["id"], c["title"])
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

# Path to the raw guidelines file (relative to the repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
GUIDELINES_PATH = _REPO_ROOT / "Knowledge" / "theinning_guidelines.txt"

# Where we cache the organized/chunked version as JSON for quick inspection
# or for tools that would rather read JSON than parse the .txt every time.
CHUNKS_CACHE_PATH = _REPO_ROOT / "Knowledge" / "knowledge_base_chunks.json"

# Sections are separated by lines of "====...." with a "N. TITLE" line
# right after the first separator, e.g.:
#
# ==================================================
# 4. COLOR SENSOR
# ==================================================
_SECTION_PATTERN = re.compile(
    r"={10,}\n(\d+)\.\s+(.+?)\n={10,}\n",
    re.MULTILINE,
)


@dataclass
class KnowledgeChunk:
    """A single retrievable unit of the knowledge base."""

    id: str            # e.g. "kb-04"
    section_number: int  # e.g. 4
    title: str          # e.g. "COLOR SENSOR"
    text: str           # the section's body text, cleaned up
    source: str = "theinning_guidelines.txt"

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(text: str) -> str:
    """Collapse extra blank lines and trim whitespace."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    return text


def organize_knowledge_base(raw_text: str) -> List[KnowledgeChunk]:
    """Split the raw guidelines text into a list of KnowledgeChunk objects.

    Any text before the first numbered section (title/purpose/disclaimer
    block at the top of the file) is kept as chunk "kb-00" so nothing is
    silently dropped.
    """
    matches = list(_SECTION_PATTERN.finditer(raw_text))
    chunks: List[KnowledgeChunk] = []

    if not matches:
        # Fallback: no recognizable section headers, treat whole file as one chunk.
        return [
            KnowledgeChunk(
                id="kb-00",
                section_number=0,
                title="FULL DOCUMENT",
                text=_clean(raw_text),
            )
        ]

    # Chunk 0: everything before the first section header (intro/disclaimer).
    intro = raw_text[: matches[0].start()]
    intro_clean = _clean(intro)
    if intro_clean:
        chunks.append(
            KnowledgeChunk(
                id="kb-00",
                section_number=0,
                title="INTRODUCTION AND DISCLAIMER",
                text=intro_clean,
            )
        )

    # Remaining chunks: one per numbered section.
    for i, m in enumerate(matches):
        section_number = int(m.group(1))
        title = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = _clean(raw_text[body_start:body_end])

        chunks.append(
            KnowledgeChunk(
                id=f"kb-{section_number:02d}",
                section_number=section_number,
                title=title,
                text=body,
            )
        )

    return chunks


def load_knowledge_base(path: Path = GUIDELINES_PATH) -> List[KnowledgeChunk]:
    """Read the guidelines file from disk and return organized chunks."""
    raw_text = path.read_text(encoding="utf-8")
    return organize_knowledge_base(raw_text)


def save_chunks_cache(chunks: List[KnowledgeChunk], path: Path = CHUNKS_CACHE_PATH) -> None:
    """Persist the organized chunks as JSON, for quick inspection/debugging
    or for other tools/agents that prefer reading structured JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [c.to_dict() for c in chunks]
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    chunks = load_knowledge_base()
    save_chunks_cache(chunks)
    print(f"Organized {len(chunks)} knowledge chunks from {GUIDELINES_PATH.name}")
    for c in chunks:
        preview = c.text.replace("\n", " ")[:70]
        print(f"  {c.id}  [{c.section_number:>2}]  {c.title:<35}  {preview}...")
