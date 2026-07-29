"""
Preprocessing with structure-aware hybrid chunking for LF Jobs RAG.

Strategy (driven by dataset analysis):
  1. Parse HTML and treat <br><b>/<strong> (and real h1-h6) as pseudo-headings.
  2. Drop noise headers (#LI-*, emails, ReqID, etc.) and pure EEO/legal boilerplate.
  3. Emit one chunk per meaningful section when size is reasonable.
  4. Fall back to sentence + size splitting (and optional semantic split) for
     large sections or jobs with no usable headings.
  5. Prefix every chunk with rich job metadata so retrieval stays job-centric.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from typing import List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from llama_index.core.node_parser import (
    HTMLNodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
)
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ------------------------------------------------------------
# Config (sourced from rag_core.config)
# ------------------------------------------------------------
from rag_core.config import (
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    CHUNK_OVERLAP_CHARS,
    CHUNK_TARGET_CHARS,
    DATA_PATH,
)
from rag_core.config import (
    SEMANTIC_EMBED_MODEL as EMBED_MODEL,
)

# Pseudo-heading tags we promote to section boundaries
HEADING_TAGS_RE = re.compile(
    r"(?:<br\s*/?\s*>\s*)*(?:<p[^>]*>\s*)?"
    r"<(b|strong|h[1-6])(?:\s+[^>]*)?>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)

# Headers that are almost always noise in this dataset
NOISE_HEADER_RE = re.compile(
    r"(?i)^("
    r"#li[-_].*"
    r"|reqid.*"
    r"|go\.[a-z0-9./-]+"
    r"|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}"
    r"|relocation (assistance|package).*"
    r"|nearest major market.*"
    r"|work shift.*"
    r"|additional locations?"
    r"|fair chance.*"
    r"|requisition code.*"
    r"|job id.*"
    r"|posted date.*"
    r")$"
)

# Strong EEO / legal signals – used only for density filtering
BOILERPLATE_KEYWORDS = [
    "equal opportunity employer",
    "equal employment opportunity",
    "affirmative action",
    "all qualified applicants",
    "without regard to",
    "regard to race",
    "race, color, religion",
    "sexual orientation",
    "gender identity",
    "protected veteran",
    "disability status",
    "reasonable accommodat",
    "eeo",
    "m/f/d/v",
    "m/f/veteran",
    "dodd frank",
    "truth in lending",
    "nmls registration",
    "criminal conviction history",
    "credit report",
    "we are committed to creating an accessible",
    "applicants will receive consideration",
]

# Normalize frequent heading variants → canonical name
HEADER_NORMALIZATION = {
    "responsibilities": "Responsibilities",
    "key responsibilities": "Responsibilities",
    "job responsibilities": "Responsibilities",
    "what you'll do": "Responsibilities",
    "what you will do": "Responsibilities",
    "what you'll be doing": "Responsibilities",
    "the day-to-day": "Responsibilities",
    "day-to-day": "Responsibilities",
    "do": "Responsibilities",
    "role purpose": "Role Overview",
    "the role": "Role Overview",
    "position summary": "Role Overview",
    "job summary": "Role Overview",
    "job description": "Role Overview",
    "overview": "Role Overview",
    "about the role": "Role Overview",
    "qualifications": "Qualifications",
    "requirements": "Qualifications",
    "minimum qualifications": "Qualifications",
    "required qualifications": "Qualifications",
    "basic qualifications": "Qualifications",
    "preferred qualifications": "Preferred Qualifications",
    "nice to have": "Preferred Qualifications",
    "nice-to-have": "Preferred Qualifications",
    "what we look for in you": "Qualifications",
    "what you'll bring": "Qualifications",
    "what you bring": "Qualifications",
    "skills": "Skills",
    "required skills": "Skills",
    "about us": "About",
    "about the team": "About",
    "about the company": "About",
    "who we are": "About",
    "who you are": "About",
    "benefits": "Benefits",
    "what we offer": "Benefits",
    "why work with us": "Benefits",
    "why join us": "Benefits",
    "compensation": "Benefits",
    "pay range": "Benefits",
    "pay transparency": "Benefits",
}


@dataclass
class JobChunk:
    chunk_id: str
    job_id: str
    chunk_index: int
    text: str
    job_title: str
    company_name: str
    job_category: str
    job_level: str
    job_location: str
    publication_date: str
    tags: str


# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
def load_jobs_dataframe(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw job listings spreadsheet into a DataFrame."""
    df = pd.read_excel(path)
    df = df.fillna("")
    return df


# ------------------------------------------------------------
# 2. HTML → Nodes (LlamaIndex)  – kept for compatibility / optional use
# ------------------------------------------------------------
def get_html_parser() -> HTMLNodeParser:
    """
    HTMLNodeParser extracts structured nodes from HTML.
    We include common tags that appear in job descriptions.
    Note: <b>/<strong> are inline; real structure is recovered
    by the custom section extractor in build_job_chunks.
    """
    return HTMLNodeParser(
        tags=["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "div", "section", "ul", "ol"]
    )


# ------------------------------------------------------------
# 3. Semantic Splitter
# ------------------------------------------------------------
def get_semantic_splitter() -> SemanticSplitterNodeParser:
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    return SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=90,  # conservative
        embed_model=embed_model,
    )


# ------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------
def _clean_html_fragment(html: str) -> str:
    """Strip tags, unescape entities, collapse whitespace."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_header(raw: str) -> str:
    key = re.sub(r"\s+", " ", raw).strip(" :.-•").lower()
    return HEADER_NORMALIZATION.get(key, raw.strip(" :.-•")[:80])


def _is_noise_header(header: str) -> bool:
    h = header.strip()
    if len(h) < 2 or len(h) > 100:
        return True
    if NOISE_HEADER_RE.match(h):
        return True
    # pure numeric / code-like
    if re.fullmatch(r"[\d\s\-_/]+", h):
        return True
    return False


def _is_boilerplate(text: str) -> bool:
    """True when the text is dominated by EEO / legal language."""
    if not text or len(text) < 40:
        return False
    lower = text.lower()
    hits = sum(1 for kw in BOILERPLATE_KEYWORDS if kw in lower)
    words = max(len(lower.split()), 1)
    # strong signal: ≥2 keywords, or ≥1 keyword in a short legal-heavy block
    if hits >= 2:
        return True
    if hits >= 1 and words < 80:
        return True
    # density check for longer blocks
    if hits >= 1 and (hits / max(words / 40, 1)) > 0.4:
        return True
    return False


def _extract_sections(html: str) -> List[Tuple[str, str]]:
    """
    Turn <br><b>…</b>, <strong>, and real heading tags into (heading, content) pairs.
    Returns at least one pair (empty heading + full cleaned text) when no headings found.
    """
    if not html or not html.strip():
        return []

    raw_matches: List[Tuple[int, int, str]] = []
    for m in HEADING_TAGS_RE.finditer(html):
        tag_text = _clean_html_fragment(m.group(2))
        if tag_text and not _is_noise_header(tag_text):
            raw_matches.append((m.start(), m.end(), tag_text))

    if not raw_matches:
        cleaned = _clean_html_fragment(html)
        return [("", cleaned)] if cleaned else []

    # Merge headings that sit very close together (common in this dataset)
    merged: List[Tuple[int, int, str]] = []
    for start, end, heading in raw_matches:
        if merged and (start - merged[-1][1]) < 40:
            prev_s, prev_e, prev_h = merged[-1]
            merged[-1] = (prev_s, end, f"{prev_h} {heading}")
        else:
            merged.append((start, end, heading))

    sections: List[Tuple[str, str]] = []
    # Leading content before the first heading
    first_start = merged[0][0]
    if first_start > 0:
        intro = _clean_html_fragment(html[:first_start])
        if intro and len(intro) >= 40 and not _is_boilerplate(intro):
            sections.append(("", intro))

    for i, (start, end, heading) in enumerate(merged):
        next_start = merged[i + 1][0] if i + 1 < len(merged) else len(html)
        content = _clean_html_fragment(html[end:next_start])
        if not content or len(content) < 30:
            continue
        if _is_boilerplate(content) and _is_boilerplate(heading + " " + content):
            continue
        sections.append((_normalize_header(heading), content))

    if not sections:
        cleaned = _clean_html_fragment(html)
        return [("", cleaned)] if cleaned else []
    return sections


def _size_split(text: str, target: int = CHUNK_TARGET_CHARS) -> List[str]:
    """Deterministic sentence-aware size split with light overlap."""
    if len(text) <= CHUNK_MAX_CHARS:
        return [text]

    splitter = SentenceSplitter(
        chunk_size=max(target // 4, 128),  # LlamaIndex uses tokens ≈ chars/4
        chunk_overlap=max(CHUNK_OVERLAP_CHARS // 4, 20),
    )
    nodes = splitter.get_nodes_from_documents([Document(text=text)])
    pieces = [n.get_content().strip() for n in nodes if n.get_content().strip()]
    return pieces if pieces else [text]


def _make_header(row: pd.Series, section: str = "") -> str:
    parts = [
        f"Job ID: {row['ID']}",
        f"Title: {row['Job Title']}",
        f"Company: {row['Company Name']}",
        f"Location: {row['Job Location']}",
        f"Level: {row['Job Level']}",
        f"Category: {row['Job Category']}",
    ]
    if section:
        parts.append(f"Section: {section}")
    return " | ".join(parts) + "\n\n"


# ------------------------------------------------------------
# 4. Build chunks
# ------------------------------------------------------------
def build_job_chunks(df: pd.DataFrame) -> List[JobChunk]:
    """
    Hybrid structure-aware chunker.

    Pipeline per job:
      1. Extract pseudo-sections from HTML (br+b / strong / h*).
      2. Drop noise headers and boilerplate sections.
      3. Emit section chunks when they fit the size budget.
      4. Size-split (sentence-aware) oversized sections.
      5. If almost no structure exists, fall back to full-text size split
         (semantic splitter can be layered on top later if desired).
      6. Prefix every chunk with rich job metadata.
    """
    # Keep references so the original function names stay meaningful
    _ = get_html_parser()
    # Semantic splitter is available for optional deeper splitting of
    # very large undifferentiated blobs; primary path is structure + size.
    try:
        semantic_splitter = get_semantic_splitter()
    except Exception:
        semantic_splitter = None

    all_chunks: List[JobChunk] = []

    for _, row in df.iterrows():
        raw_html = str(row.get("Job Description", "") or "")
        if not raw_html.strip():
            continue

        sections = _extract_sections(raw_html)

        # Fallback path: no useful sections → treat whole description as one block
        if len(sections) == 1 and sections[0][0] == "":
            full = sections[0][1]
            if _is_boilerplate(full):
                # Still emit a minimal metadata-only style chunk so the job is findable
                pieces = [full[:CHUNK_TARGET_CHARS]]
            elif len(full) <= CHUNK_MAX_CHARS:
                pieces = [full]
            else:
                # Prefer size split; optionally refine with semantic if available
                pieces = _size_split(full)
                if semantic_splitter and len(pieces) == 1 and len(full) > CHUNK_MAX_CHARS * 1.5:
                    try:
                        sem_nodes = semantic_splitter.get_nodes_from_documents(
                            [Document(text=full)]
                        )
                        sem_pieces = [
                            n.get_content().strip()
                            for n in sem_nodes
                            if n.get_content().strip()
                        ]
                        if sem_pieces:
                            # Re-cap any still-oversized semantic pieces
                            pieces = []
                            for sp in sem_pieces:
                                pieces.extend(_size_split(sp) if len(sp) > CHUNK_MAX_CHARS else [sp])
                    except Exception:
                        pass
            section_label = ""
            text_pieces = [(section_label, p) for p in pieces]
        else:
            text_pieces = []
            for heading, content in sections:
                if len(content) <= CHUNK_MAX_CHARS:
                    text_pieces.append((heading, content))
                else:
                    for sub in _size_split(content):
                        text_pieces.append((heading, sub))

        # Assemble final JobChunk objects
        chunk_idx = 0
        pending_tiny: List[Tuple[str, str]] = []

        def flush_pending(into_text: str = "") -> str:
            nonlocal pending_tiny
            if not pending_tiny:
                return into_text
            extra = "\n\n".join(
                (f"[{h}] {t}" if h else t) for h, t in pending_tiny
            )
            pending_tiny = []
            return (into_text + "\n\n" + extra).strip() if into_text else extra

        for heading, piece in text_pieces:
            piece = piece.strip()
            if not piece:
                continue

            # Collect tiny fragments and merge into the next real chunk
            if len(piece) < CHUNK_MIN_CHARS:
                pending_tiny.append((heading, piece))
                continue

            body = flush_pending()
            if body:
                body = body + "\n\n"
            if heading:
                body += f"[{heading}]\n{piece}"
            else:
                body += piece

            header = _make_header(row, section=heading)
            full_text = header + body

            all_chunks.append(
                JobChunk(
                    chunk_id=f"{row['ID']}_chunk{chunk_idx}",
                    job_id=str(row["ID"]),
                    chunk_index=chunk_idx,
                    text=full_text,
                    job_title=str(row["Job Title"]),
                    company_name=str(row["Company Name"]),
                    job_category=str(row["Job Category"]),
                    job_level=str(row["Job Level"]),
                    job_location=str(row["Job Location"]),
                    publication_date=str(row["Publication Date"]),
                    tags=str(row.get("Tags", "")),
                )
            )
            chunk_idx += 1

        # Trailing tiny fragments → attach to last chunk or emit alone
        if pending_tiny:
            extra = flush_pending()
            if all_chunks and all_chunks[-1].job_id == str(row["ID"]):
                all_chunks[-1].text += "\n\n" + extra
            else:
                header = _make_header(row)
                all_chunks.append(
                    JobChunk(
                        chunk_id=f"{row['ID']}_chunk{chunk_idx}",
                        job_id=str(row["ID"]),
                        chunk_index=chunk_idx,
                        text=header + extra,
                        job_title=str(row["Job Title"]),
                        company_name=str(row["Company Name"]),
                        job_category=str(row["Job Category"]),
                        job_level=str(row["Job Level"]),
                        job_location=str(row["Job Location"]),
                        publication_date=str(row["Publication Date"]),
                        tags=str(row.get("Tags", "")),
                    )
                )

    return all_chunks
