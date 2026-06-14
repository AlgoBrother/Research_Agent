from datetime import datetime
from typing import List, Dict
from pathlib import Path

from models.classes import Paper
from models.llm import chat

SYSTEM = (
    "You are a senior  technica; research analyst"
     "Write clearly and precisely. No filler. No hype."
)


REPORT_PROMPT = """You are synthesizing recent research papers into a technical digest.
 
User's original query: {query}
Topic: {topic}
 
Papers (ranked by relevance, most relevant first):
{evidence}
 
Write a research digest with these sections:
 
## Key Findings
3-5 bullet points covering the most important insights across the papers.
 
## Recommendations
What should the user do or investigate next, given their query?
 
## Tradeoffs / Open Questions
What are the limitations, disagreements between papers, or unresolved questions?
 
## Sources
List each paper as: [Title](url) — one-line description
 
Be specific and technical. Reference paper titles when making claims."""
 
def _format_evidence(papers: List[Paper]) -> str:
    """This function writes evidences for our generated report

    Args:
        papers (List[Paper]): _description_

    Returns:
        str: _description_
    """
    lines = []
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        lines.append(
            f"[{i}] {p.title}\n"
            f"    Authors: {authors} | Published: {p.published.strftime('%Y-%m-%d')}\n"
            f"    Score: {p.relevance_score:.2f}\n"
            f"    Abstract: {p.summary[:400]}...\n"
            f"    URL: {p.pdf_url}"
        )
    return "\n\n".join(lines)

def generate_report(query: str, topic: str, papers: List[Paper], model: str = "llama-3.3-70b-versatile") -> str: # bigger model for better synthesis
    evidence = _format_evidence(papers)
    prompt = REPORT_PROMPT.format(query=query, topic=topic, evidence=evidence)
    return chat(prompt, system=SYSTEM, model=model, max_tokens=1500, temperature=0.3)

def save_report(query: str, report: str, papers: List[Paper]) -> Path:
    """Write markdown report to disk. Same format as your original."""
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    slug = query[:30].replace(" ", "_").replace("/", "-")
    filename = Path(f"research_digest_{date_str}_{slug}.md")
 
    with filename.open("w") as f:
        f.write(f"# Research Digest\n")
        f.write(f"**Query:** {query}\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("---\n\n")
        f.write(report)
        f.write("\n\n---\n\n")
        f.write("## Raw Paper Index\n\n")
        for p in papers:
            f.write(f"- [{p.title}]({p.pdf_url}) — {p.published.strftime('%Y-%m-%d')}\n")
 
    print(f"✅ Report saved → {filename}")
    return filename
 

 
