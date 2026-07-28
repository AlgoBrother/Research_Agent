import re 
from typing import List, Set
from models.classes import PooledChunk

class ToonCompiler:
    """
    Compiles a normalized list of PooledChunk objects into 
    Token-Oriented Object Notation (TOON) for prompt context.
    """

    def compile(self, chunks: List[PooledChunk]) -> str:
        if not chunks:
            return "context_chunks[0]{source,source_id,trust_tag,score,title,section,text}:"

        # 1. Deduplicate chunks across sources (e.g. same arXiv paper returned by Web & arXiv)
        deduped: List[PooledChunk] = []
        seen_keys: Set[str] = set()

        for chunk in chunks:
            # Uniqueness key based on source, source_id, and section
            dedup_key = f"{chunk.source}:{chunk.source_id}:{chunk.section}:{hash(chunk.text)}"
            if dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                deduped.append(chunk)

        # 2. Declare TOON schema header once
        header = f"context_chunks[{len(deduped)}]{{source,source_id,trust_tag,score,title,section,text}}:"
        rows = [header]

        # 3. Format each PooledChunk into a CSV-like TOON row
        for c in deduped:
            s_source = c.source.value if hasattr(c.source, "value") else str(c.source)
            s_id = self._clean_field(c.source_id)
            s_tag = c.trust_tag.value if hasattr(c.trust_tag, "value") else str(c.trust_tag)
            s_score = f"{c.score:.2f}"
            s_title = self._clean_field(c.title)
            s_section = self._clean_field(c.section)
            s_text = self._clean_field(c.text)

            row = f'{s_source},{s_id},{s_tag},{s_score},"{s_title}","{s_section}","{s_text}"'
            rows.append(row)

        return "\n".join(rows)

    def _clean_field(self, text: str) -> str:
        """Sanitizes text strings by stripping newlines and escaping double quotes."""
        if not text:
            return ""
        # Collapse multi-line text into a single space-separated line
        cleaned = re.sub(r"\s+", " ", text).strip()
        # Escape inner double quotes for CSV safety
        return cleaned.replace('"', '""')