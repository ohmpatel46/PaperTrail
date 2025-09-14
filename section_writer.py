#!/usr/bin/env python3
"""
section_writer.py
- Section-wise prompting with Ollama
- Inputs: user prompt, selected_papers.json (list or {"papers":[...]})
- Uses clipped context per paper (abstract + top sentences from full_text if present)
- Outputs: IEEE-style Markdown

Example:
  python section_writer.py --papers selected_papers.json \
    --prompt "Build an on-device embedding + retrieval system on Snapdragon NPU for offline research search" \
    --out draft.md --model llama3.2:3b
"""

import os, re, json, argparse, requests, textwrap, sys
from typing import List, Dict

# ---------------- Config ----------------
DEFAULT_MODEL = "llama3.2:3b"   # you can pass --model llama3.2:3b-instruct if you pull it
OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434")
MAX_TOKENS    = 450
TEMPERATURE   = 0.2

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

# ------------- Basic keywording (for clipping only) -------------
STOP = set("a an the and or for of to in on at by with from as is are was were be been this that these those it its into about over under between among within across per via your you we they he she them his her our their than then too very just also not no nor such same own each both either neither may might can could should would will shall do does did done having have has had without against during before after above below again further more most less least".split())
WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]+")

def tokenize(s: str) -> List[str]:
    return [t.lower() for t in WORD.findall(s) if t.lower() not in STOP and len(t)>=3]

def extract_keywords(prompt: str, k: int = 10) -> List[str]:
    toks = tokenize(prompt)
    # light ranking by frequency; include 2-grams
    bigrams = [" ".join(toks[i:i+2]) for i in range(len(toks)-1)]
    scores = {}
    for w in toks: scores[w] = scores.get(w,0)+1
    for b in bigrams: scores[b] = scores.get(b,0)+1.5
    # drop generic
    generic = {"research","paper","project","study","system","model","models","data","dataset"}
    ranked = [w for w,_ in sorted(scores.items(), key=lambda x:(-x[1], x[0])) if w not in generic]
    return ranked[:k]

# ------------- Clipping utilities -------------
SECTION_HINTS = ["abstract","introduction","related","background","method","approach","experiments","evaluation","results","discussion","conclusion","future work"]
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def rough_sections(full_text: str) -> Dict[str,str]:
    if not full_text: return {}
    text = full_text.replace("\r","")
    parts = re.split(r"\n\s*\n", text)
    sections = {"other":[]}
    current = "other"
    for block in parts:
        head = block.strip().lower().split("\n",1)[0][:80]
        matched = None
        for h in SECTION_HINTS:
            if re.search(rf"\b{h}\b", head): matched = h; break
        if matched:
            current = matched
            sections.setdefault(current, [])
        sections[current].append(block.strip())
    return {k: "\n\n".join(v) for k,v in sections.items() if v}

def sent_tokenize(text: str) -> List[str]:
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]

def keyword_score(sent: str, kws: List[str]) -> float:
    s = sent.lower().replace("-", " ")
    score = 0.0
    for k in kws:
        k2 = k.lower().replace("-", " ")
        if " " in k2:
            score += 2.0 * s.count(k2)
            w1, *rest = k2.split()
            score += 0.5 * len(re.findall(rf"\b{re.escape(w1)}\b", s))
            if rest:
                score += 0.5 * len(re.findall(rf"\b{re.escape(rest[0])}\b", s))
        else:
            score += 1.0 * len(re.findall(rf"\b{re.escape(k2)}\b", s))
    return score

def clip_paper_digest(p: dict, kws: List[str], per_paper_chars: int = 1600, max_sents_per_section: int = 4) -> str:
    title = (p.get("title") or "").strip()
    year  = p.get("year","")
    abstract = (p.get("abstract") or "").strip()
    lines = [f"- {title} ({year}) :: {abstract[:700]}{'...' if len(abstract)>700 else ''}"]
    full_text = (p.get("full_text") or "").strip()
    if full_text:
        sec_map = rough_sections(full_text)
        order = ["introduction","method","approach","experiments","results","discussion","conclusion","related","background","other"]
        used = len("\n".join(lines))
        for sec in order:
            if sec not in sec_map: continue
            sents = sent_tokenize(sec_map[sec])
            scored = sorted(((keyword_score(s,kws), s) for s in sents), key=lambda x: -x[0])
            top = [s for sc,s in scored[:max_sents_per_section] if sc>0]
            if not top: continue
            blob = " ".join(top)
            chunk = f"• [{sec.title()}] {blob}"
            if used + len(chunk) > per_paper_chars:
                chunk = chunk[:max(0, per_paper_chars - used)]
            if chunk:
                lines.append(chunk)
                used += len(chunk)
            if used >= per_paper_chars: break
    return "\n".join(lines)

def build_context(selected: List[dict], prompt: str, global_budget: int = 8000, per_paper: int = 1600) -> str:
    kws = extract_keywords(prompt, k=10)
    chunks, total = [], 0
    for p in selected:
        d = clip_paper_digest(p, kws, per_paper_chars=per_paper)
        if not d: continue
        if total + len(d) > global_budget:
            d = d[:max(0, global_budget-total)]
        if not d: break
        chunks.append(d); total += len(d)
        if total >= global_budget: break
    return "\n\n".join(chunks)

# ------------- Ollama helper -------------
def ollama_complete(system: str, user: str, model: str, url: str, max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE, timeout: int = 120) -> str:
    prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
    r = requests.post(f"{url.rstrip('/')}/api/generate",
                      json={"model": model, "prompt": prompt, "stream": False,
                            "options":{"temperature": temperature, "num_predict": max_tokens, "num_ctx": 4096}},
                      timeout=timeout)
    if r.status_code == 404:
        try: msg = r.json().get("error","model not found")
        except Exception: msg = "model not found"
        raise RuntimeError(f"Ollama 404: {msg}. Try `ollama pull {model}`.")
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "response" in data:
        return data["response"].strip()
    raise RuntimeError(f"Unexpected Ollama response: {data}")

# ------------- Section prompts -------------
def gen_abstract(prompt: str, model: str, url: str) -> str:
    sysmsg = "Write a 3–5 sentence IEEE-style ABSTRACT summarizing the user's research goal. Be concise and factual; no citations."
    user = f"User goal:\n---\n{prompt}\n---"
    return ollama_complete(sysmsg, user, model, url, max_tokens=220)

def gen_intro(context: str, prompt: str, model: str, url: str) -> str:
    sysmsg = ("Write the 'Introduction' section (1–2 paragraphs). "
              "Motivate the problem, define scope, and briefly preview a plausible approach. "
              "Base on the provided paper digests; do not fabricate specific results.")
    user = f"User goal:\n{prompt}\n\nPaper digests:\n{context}"
    return ollama_complete(sysmsg, user, model, url, max_tokens=350)

def gen_related(context: str, model: str, url: str) -> str:
    sysmsg = ("Write 'Related Work' as a compact paragraph or 5–8 bullet points. "
              "Group ideas by themes; avoid numeric claims. No citations formatting (we'll add later).")
    user = f"Paper digests:\n{context}"
    return ollama_complete(sysmsg, user, model, url, max_tokens=320)

def gen_method(context: str, prompt: str, model: str, url: str) -> str:
    sysmsg = ("Write the 'Method' section at a high level (2–4 paragraphs). "
              "Outline components, data flow, and key decisions relevant to the user's goal. "
              "Stay generic; no invented metrics or results.")
    user = f"User goal:\n{prompt}\n\nPaper digests:\n{context}"
    return ollama_complete(sysmsg, user, model, url, max_tokens=420)

def gen_experiments(context: str, model: str, url: str) -> str:
    sysmsg = ("Write 'Experiments' describing datasets (examples), training/eval setup, and metrics likely used. "
              "Do not fabricate numbers; provide placeholders where needed.")
    user = f"Paper digests:\n{context}"
    return ollama_complete(sysmsg, user, model, url, max_tokens=350)

def gen_results(context: str, model: str, url: str) -> str:
    sysmsg = ("Write 'Results' focusing on how you would analyze outcomes (ablation ideas, latency/accuracy trade-offs). "
              "No numeric claims. Mention qualitative/quantitative analysis approaches.")
    user = f"Paper digests:\n{context}"
    return ollama_complete(sysmsg, user, model, url, max_tokens=280)

def gen_conclusion(context: str, model: str, url: str) -> str:
    sysmsg = ("Write 'Conclusion' (2–4 sentences) summarizing contributions and next steps. "
              "Avoid strong claims without evidence.")
    user = f"Paper digests:\n{context}"
    return ollama_complete(sysmsg, user, model, url, max_tokens=200)

# ------------- IO -------------
def load_selected(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f: d = json.load(f)
    if isinstance(d, dict) and "papers" in d: d = d["papers"]
    if not isinstance(d, list): raise ValueError("selected papers JSON must be a list or {'papers':[...]}")
    return d

# ------------- Main -------------
def main():
    ap = argparse.ArgumentParser(description="Section-wise IEEE draft with clipped paper context (Ollama).")
    ap.add_argument("--papers", required=True, help="JSON of selected papers (list or {'papers':[...]}). Supports optional 'full_text'.")
    ap.add_argument("--prompt", required=True, help="User research idea.")
    ap.add_argument("--title", default="Draft Title", help="Paper title.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    ap.add_argument("--ollama-url", default=OLLAMA_URL, help="Ollama base URL.")
    ap.add_argument("--out", default="draft.md", help="Output Markdown file.")
    ap.add_argument("--global-budget", type=int, default=8000, help="Total chars budget for LLM context.")
    ap.add_argument("--per-paper", type=int, default=1600, help="Per-paper chars budget in context.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    papers = load_selected(args.papers)
    if args.debug:
        print(f"Loaded {len(papers)} papers")

    context = build_context(papers, args.prompt, global_budget=args.global_budget, per_paper=args.per_paper)
    if args.debug:
        print(f"Context chars: {len(context)}")

    sections = {}
    sections["TITLE"] = args.title
    sections["ABSTRACT"] = gen_abstract(args.prompt, args.model, args.ollama_url)
    sections["INTRO"] = gen_intro(context, args.prompt, args.model, args.ollama_url)
    sections["RELATED"] = gen_related(context, args.model, args.ollama_url)
    sections["METHOD"] = gen_method(context, args.prompt, args.model, args.ollama_url)
    sections["EXPERIMENTS"] = gen_experiments(context, args.model, args.ollama_url)
    sections["RESULTS"] = gen_results(context, args.model, args.ollama_url)
    sections["CONCLUSION"] = gen_conclusion(context, args.model, args.ollama_url)
    sections["REFERENCES"] = ""  # will be filled by Auto-Cite later

    md = IEEE_TEMPLATE.format(**sections)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {args.out} (model={args.model})")

if __name__ == "__main__":
    main()
