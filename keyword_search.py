#!/usr/bin/env python3
import os, re, json, math, sqlite3, argparse, sys
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

STOPWORDS = set("""
a an the and or for of to in on at by with from as is are was were be been being this that these those
it its into about over under between among within across per via into your you we they he she them his her
our their i me my mine ours theirs us than then too very just also not no nor such same own each both either
neither may might can could should would will shall do does did done doing have has had having
into without against during before after above below up down out off again further more most less least
""".split())

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]+")

def tokenize(text: str) -> List[str]:
    toks = [t.lower() for t in WORD_RE.findall(text)]
    return [t for t in toks if t not in STOPWORDS and len(t) >= 3]

def ngrams(tokens: List[str], n=2):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def extract_keywords(prompt: str, max_keywords: int = 8) -> List[str]:
    toks = tokenize(prompt)
    tf = Counter(toks)
    bigrams = ngrams(toks, 2)
    tf2 = Counter([b for b in bigrams if all(w not in STOPWORDS for w in b.split())])

    scores = defaultdict(float)
    for w,c in tf.items(): scores[w] += c
    for b,c in tf2.items(): scores[b] += 1.5*c  # favor bigrams

    for b,_ in tf2.items():
        w1,w2 = b.split()
        scores[w1] *= 0.9
        scores[w2] *= 0.9

    cand = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    generic = {"research","paper","project","study","system","model","models","data","dataset"}
    out = []
    for w,_ in cand:
        if w in generic: continue
        out.append(w)
        if len(out) >= max_keywords: break
    return out

def load_papers(sqlite_path: str, json_path: str) -> Tuple[List[Dict], str]:
    if os.path.exists(sqlite_path):
        try:
            con = sqlite3.connect(sqlite_path)
            cur = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='papers'")
            if cur.fetchone():
                rows = con.execute("SELECT paper_id,title,authors,abstract,url,year FROM papers").fetchall()
                papers = []
                for r in rows:
                    papers.append({
                        "paper_id": r[0], "title": r[1], "authors": r[2],
                        "abstract": r[3], "url": r[4], "year": r[5]
                    })
                return papers, "sqlite"
        except Exception:
            pass
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f), "json"

def build_idf(docs: List[str]) -> Dict[str, float]:
    N = len(docs)
    df = Counter()
    for d in docs:
        terms = set(tokenize(d))
        df.update(terms)
    idf = {}
    for t, c in df.items():
        idf[t] = math.log((N + 1) / (c + 1)) + 1.0
    return idf

# ---------- forgiving scorer ----------
def _normalize(s: str) -> str:
    s = s.lower().replace("-", " ")
    # simple plural → singular for trailing 's'
    s = " ".join([w[:-1] if w.endswith("s") and len(w) > 3 else w for w in s.split()])
    return s

def score_paper(keywords: List[str], title: str, abstract: str, idf: Dict[str,float]) -> float:
    """
    Normalizes text (lowercase, de-hyphen, simple singularization),
    searches in title + abstract,
    bigrams get higher weight if exact phrase match; otherwise fall back to unigram hits with smaller weight.
    """
    text = _normalize(f"{title} {abstract}")
    score = 0.0
    for k in keywords:
        k_norm = _normalize(k)
        if " " in k_norm:
            occ_phrase = text.count(k_norm)
            if occ_phrase > 0:
                score += occ_phrase * 2.0 * idf.get(k_norm.split()[0], 1.0)
            else:
                parts = k_norm.split()
                part_hits = 0
                for w in parts[:2]:
                    part_hits += len([1 for _ in re.finditer(rf"\b{re.escape(w)}\b", text)])
                score += 0.6 * part_hits * idf.get(parts[0], 1.0)
        else:
            occ = len([1 for _ in re.finditer(rf"\b{re.escape(k_norm)}\b", text)])
            score += occ * 1.0 * idf.get(k_norm, 1.0)
    return score
# --------------------------------------

def find_relevant(prompt: str, papers: List[Dict], topn: int = 5, min_score: float = 0.0, debug: bool=False):
    keywords = extract_keywords(prompt, max_keywords=8)
    idf = build_idf([p["abstract"] for p in papers] + [prompt])
    scored = []
    for p in papers:
        s = score_paper(keywords, p["title"], p["abstract"], idf)
        scored.append((s, p))
        if debug:
            print(f"[DEBUG] {p['paper_id']} score={s:.3f} :: {p['title']}")
    scored.sort(key=lambda x: -x[0])

    # Filter by min_score; if all zero, fall back to topn regardless
    filtered = [(s,p) for (s,p) in scored if s >= min_score]
    if not filtered:
        filtered = scored  # graceful fallback

    return keywords, [p for _,p in filtered[:topn]]

def main():
    ap = argparse.ArgumentParser(description="Keyword extraction + simple abstract matching (JSON fallback).")
    ap.add_argument("--db", default="db.sqlite", help="Path to SQLite DB (optional).")
    ap.add_argument("--json", default="papers.json", help="Fallback JSON with sample papers.")
    ap.add_argument("--topn", type=int, default=5, help="Max results to show.")
    ap.add_argument("--min_score", type=float, default=0.0, help="Minimum score to include a paper.")
    ap.add_argument("--debug", action="store_true", help="Print per-paper scores.")
    ap.add_argument("--prompt", type=str, default=None, help="Prompt text (if omitted, read from stdin).")
    args = ap.parse_args()

    if args.prompt is None:
        print("Enter your research idea (end with Ctrl-D / Ctrl-Z):")
        try:
            prompt = sys.stdin.read().strip()
        except KeyboardInterrupt:
            print("\nAborted."); sys.exit(1)
    else:
        prompt = args.prompt.strip()

    if not prompt:
        print("No prompt provided."); sys.exit(1)

    papers, source = load_papers(args.db, args.json)
    if not papers:
        print("No papers found in DB or JSON."); sys.exit(1)

    keywords, results = find_relevant(prompt, papers, topn=args.topn, min_score=args.min_score, debug=args.debug)

    print("\n=== Top Keywords ===")
    for k in keywords:
        print(f"- {k}")

    print(f"\n=== Relevant Papers (source: {source}) ===")
    if not results:
        print("(No matches; try a different prompt.)")
    for i,p in enumerate(results,1):
        print(f"{i}. {p['title']} ({p.get('year','')})")
        print(f"   ID: {p['paper_id']}")
        print(f"   URL: {p['url']}")
        print(f"   Abstract: {p['abstract'][:200]}{'...' if len(p['abstract'])>200 else ''}")
    print("")

if __name__ == "__main__":
    main()
