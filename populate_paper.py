#!/usr/bin/env python3
"""
draft_from_selected.py
- INPUT: user prompt + a JSON file of already-filtered papers (from your prior script)
- ACTION: call a local LLM (Ollama) to synthesize IEEE-style sections
- OUTPUT: Markdown draft

Usage:
  python draft_from_selected.py --papers selected_papers.json \
    --prompt "your idea" --out draft.md \
    --model llama3.2:3b-instruct
"""

import os, re, json, argparse, sys, requests, textwrap

# ---------------- Config ----------------
DEFAULT_MODEL = "llama3.2:3b-instruct"           # Ollama model
OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MAX_TOKENS = 700
LLM_TEMPERATURE = 0.2

IEEE_TEMPLATE = """# {TITLE}

**Abstract—** {ABSTRACT}

## I. Introduction
{INTRO}

## II. Related Work
{RELATED}

## III. Method
{METHOD}

## IV. Experiments
{EXPERIMENTS}

## V. Results
{RESULTS}

## VI. Conclusion
{CONCLUSION}

## References
{REFERENCES}
"""

# -------------- LLM helpers --------------
def ollama_generate(system, user, model, base_url="http://localhost:11434",
                    max_tokens=700, temperature=0.2, timeout=120):
    import requests, json, re
    prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
    url = f"{base_url.rstrip('/')}/api/generate"
    r = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,  # <<< prevent NDJSON streaming so we can r.json()
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 4096}
        },
        timeout=timeout
    )
    if r.status_code == 404:
        # often means model not pulled
        try:
            msg = r.json().get("error", "model not found")
        except Exception:
            msg = "model not found"
        raise RuntimeError(f"Ollama 404: {msg}. Try: `ollama pull {model}`.")
    r.raise_for_status()
    data = r.json()
    # standard path
    if isinstance(data, dict) and "response" in data:
        return data["response"]
    # fallback: extract first JSON object with 'response'
    m = re.search(r"\{.*?\}", r.text, flags=re.DOTALL)
    if m:
        try:
            j = json.loads(m.group(0))
            if "response" in j:
                return j["response"]
        except Exception:
            pass
    raise RuntimeError(f"Unexpected Ollama response shape: {r.text[:200]}...")


def sections_from_llm(user_prompt: str, selected_papers: list, title_override: str | None, model: str):
    # pack concise paper context
    lines = []
    for p in selected_papers:
        title = p.get("title","").strip()
        year  = p.get("year","")
        abstract = (p.get("abstract","") or "").replace("\n"," ").strip()
        abstract = abstract[:800] + ("..." if len(abstract) > 800 else "")
        lines.append(f"- {title} ({year}) :: {abstract}")
    context = "\n".join(lines) if lines else "(no papers provided)"

    system = (
        "You draft IEEE-style sections for a research paper.\n"
        "OUTPUT STRICT JSON ONLY matching the schema provided. "
        "Do not fabricate numeric results or citations; keep a neutral tone. "
        "Base the ABSTRACT primarily on the user's prompt; infer the other sections from the provided context."
    )

    schema = """{
  "TITLE": "short, specific title (<= 14 words)",
  "ABSTRACT": "3-5 sentence abstract based on the user prompt",
  "INTRO": "1-2 paragraphs introducing the problem and motivation",
  "RELATED": "summary of related work derived from the provided paper abstracts",
  "METHOD": "proposed approach at a high level; avoid fabricated details",
  "EXPERIMENTS": "what to evaluate, plausible datasets/metrics",
  "RESULTS": "how results would be analyzed; no fake numbers",
  "CONCLUSION": "2-4 sentences with contributions and next steps",
  "REFERENCES": "leave empty; citations handled later"
}"""

    user = textwrap.dedent(f"""
    User prompt (goal/idea):
    ---
    {user_prompt}
    ---

    Selected paper abstracts (supporting context):
    ---
    {context}
    ---

    Output STRICT JSON with exactly these keys:
    {schema}
    """).strip()

    raw = ollama_generate(system, user, model=model)
    # parse JSON (extract a JSON block if needed)
    data = None
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                pass
    if not isinstance(data, dict):
        # fallback minimal sections
        data = {
            "TITLE": title_override or "Draft Title",
            "ABSTRACT": f"This paper explores: {user_prompt}",
            "INTRO": "Introduction draft.",
            "RELATED": "Related work draft.",
            "METHOD": "Method draft.",
            "EXPERIMENTS": "Experiments draft.",
            "RESULTS": "Results draft.",
            "CONCLUSION": "Conclusion draft.",
            "REFERENCES": ""
        }
    # ensure all keys exist & strip
    keys = ["TITLE","ABSTRACT","INTRO","RELATED","METHOD","EXPERIMENTS","RESULTS","CONCLUSION","REFERENCES"]
    for k in keys:
        data[k] = (data.get(k) or "").strip()
    if title_override:
        data["TITLE"] = title_override.strip()
    return data

def render_markdown(sections: dict) -> str:
    return IEEE_TEMPLATE.format(**sections)

# -------------- IO helpers --------------
def load_selected_papers(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # accept either {"papers":[...]} or bare list [...]
    if isinstance(data, dict) and "papers" in data:
        data = data["papers"]
    if not isinstance(data, list):
        raise ValueError("Selected papers JSON must be a list of paper objects or an object with a 'papers' list.")
    # minimal fields
    for p in data:
        for k in ("title","abstract"):
            if k not in p:
                raise ValueError(f"Paper missing required field '{k}': {p}")
    return data

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="Draft IEEE-style paper from already-selected papers + user prompt.")
    ap.add_argument("--papers", required=True, help="Path to JSON with selected papers (list or {'papers': [...]})")
    ap.add_argument("--prompt", required=True, help="User research idea/prompt.")
    ap.add_argument("--title", default=None, help="Optional manual title.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    ap.add_argument("--out", default="draft.md", help="Output Markdown file.")
    ap.add_argument("--debug", action="store_true", help="Print debug info.")
    args = ap.parse_args()

    selected = load_selected_papers(args.papers)
    if args.debug:
        print(f"Loaded {len(selected)} selected papers from {args.papers}")
        for p in selected[:5]:
            print("-", p.get("title","")[:90])

    sections = sections_from_llm(args.prompt, selected, title_override=args.title, model=args.model)
    md = render_markdown(sections)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {args.out} (model={args.model}, papers={len(selected)})")

if __name__ == "__main__":
    main()
