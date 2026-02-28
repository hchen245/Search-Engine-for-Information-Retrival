import argparse
import json
import math
import os
import re
from collections import defaultdict

from nltk.stem import PorterStemmer


"""Milestone 2 retrieval component.

Features:
1) Boolean AND-only retrieval
2) Optional tf-idf ranking for matched docs
3) Text-based interactive search interface
4) One-command execution for the 4 required milestone queries
"""


DATA_PATH = "DEV"
PARTIAL_INDEX_DIR = "partial_indexes"
FINAL_INDEX_DIR = "final_index"
FINAL_INDEX_FILE = os.path.join(FINAL_INDEX_DIR, "final_index.txt")
DOC_MAP_PATH = os.path.join(FINAL_INDEX_DIR, "doc_id_map.json")

MILESTONE_QUERIES = [
    "cristina lopes",
    "machine learning",
    "ACM",
    "master of software engineering",
]

stemmer = PorterStemmer()


def strip_fragment(url):
    """Return URL without fragment part (#...)."""
    if not isinstance(url, str):
        return ""
    return url.split("#", 1)[0]


def tokenize(text):
    """Split raw text into lowercase alphanumeric tokens."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def stem_tokens(tokens):
    """Apply Porter stemming to each token."""
    return [stemmer.stem(token) for token in tokens]


def normalize_query(query):
    """Normalize query by tokenizing and stemming (no stopword removal)."""
    tokens = tokenize(query)
    return stem_tokens(tokens)


def build_doc_id_map_if_missing():
    """Load doc_id -> URL map from disk, or build it from DEV if missing.

    The mapping is needed because postings store doc IDs while report output
    requires URLs.
    """
    if os.path.exists(DOC_MAP_PATH):
        with open(DOC_MAP_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(doc_id): strip_fragment(url) for doc_id, url in raw.items()}

    os.makedirs(FINAL_INDEX_DIR, exist_ok=True)
    doc_id_to_url = {}
    doc_id = 1

    print("Building doc_id -> URL map from DEV/... (one-time)")
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            if not file.endswith(".json"):
                continue

            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                doc_id_to_url[doc_id] = strip_fragment(data.get("url", ""))
            except Exception:
                doc_id_to_url[doc_id] = ""
            doc_id += 1

    with open(DOC_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(doc_id_to_url, f, ensure_ascii=False)

    print(f"Saved {len(doc_id_to_url)} URL mappings to {DOC_MAP_PATH}")
    return doc_id_to_url


def parse_postings_line(line):
    """Parse a postings line: 'term: docID1 tf1 docID2 tf2 ...'."""
    term, postings_str = line.strip().split(":", 1)
    entries = postings_str.strip().split()
    postings = {}

    if len(entries) % 2 != 0:
        return term, postings

    for i in range(0, len(entries), 2):
        try:
            doc_id = int(entries[i])
            tf = int(entries[i + 1])
            postings[doc_id] = tf
        except ValueError:
            continue

    return term, postings # {doc_id: tf, ...}


def load_query_postings(query_terms):
    """Load postings only for query terms by scanning partial indexes.

    This avoids loading the full inverted index into memory.
    """
    needed_terms = set(query_terms)
    merged = {term: defaultdict(int) for term in needed_terms}

    # Prefer merged final index if available.
    if os.path.exists(FINAL_INDEX_FILE):
        with open(FINAL_INDEX_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue

                term = line.split(":", 1)[0]
                if term not in needed_terms:
                    continue

                _, postings = parse_postings_line(line)
                for doc_id, tf in postings.items():
                    merged[term][doc_id] += tf
        return {term: dict(doc_tf) for term, doc_tf in merged.items()}

    if not os.path.isdir(PARTIAL_INDEX_DIR):
        raise FileNotFoundError(
            "No index files found. Run `python indexer.py` to generate final_index/final_index.txt."
        )

    for filename in sorted(os.listdir(PARTIAL_INDEX_DIR)):
        if not filename.startswith("partial_"):
            continue

        file_path = os.path.join(PARTIAL_INDEX_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue

                term = line.split(":", 1)[0]
                if term not in needed_terms:
                    continue

                _, postings = parse_postings_line(line)
                # Merge postings from all partial files for the same term.
                for doc_id, tf in postings.items():
                    merged[term][doc_id] += tf

    return {term: dict(doc_tf) for term, doc_tf in merged.items()}


def and_search(query, doc_id_map, top_k=5):
    """Run Boolean AND retrieval and return top_k ranked URLs.

    Ranking is tf-idf on the boolean-filtered candidate set.
    """
    terms = normalize_query(query)
    if not terms:
        return []

    postings_by_term = load_query_postings(terms)
    term_docs = []

    for term in terms:
        postings = postings_by_term.get(term, {})
        if postings:
            term_docs.append(set(postings.keys()))
        else:
            # If any term is missing, AND query has no results.
            return []

    if not term_docs:
        return []

    candidate_docs = set.intersection(*term_docs)

    if not candidate_docs:
        return []

    total_docs = len(doc_id_map)
    idf = {}
    for term in terms:
        df = len(postings_by_term[term])
        # Smoothed IDF.
        idf[term] = math.log((total_docs + 1) / (df + 1)) + 1.0

    scored = []
    for doc_id in candidate_docs:
        score = 0.0
        for term in terms:
            tf = postings_by_term[term].get(doc_id, 0)
            if tf > 0:
                # Log-scaled TF.
                score += (1 + math.log(tf)) * idf[term]
        scored.append((doc_id, score))

    scored.sort(key=lambda item: (-item[1], item[0]))

    results = []
    seen_urls = set()
    for doc_id, score in scored:
        url = doc_id_map.get(doc_id, "")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        results.append(
            {
                "doc_id": doc_id,
                "url": url,
                "score": round(score, 6),
            }
        )
        if len(results) >= top_k:
            break

    return results


def run_milestone_queries(doc_id_map, top_k=5):
    """Execute the 4 required milestone queries and print top results."""
    all_results = {}
    for i, query in enumerate(MILESTONE_QUERIES, start=1):
        print("=" * 80)
        print(f"Query {i}: {query}")
        results = and_search(query, doc_id_map, top_k=top_k)
        all_results[query] = results

        if not results:
            print("No results found.")
            continue

        for rank, item in enumerate(results, start=1):
            print(f"{rank}. {item['url']}  (doc_id={item['doc_id']}, score={item['score']})")

    return all_results


def interactive_mode(doc_id_map, top_k=5):
    """Start simple CLI loop for manual query testing."""
    print("Search interface started. Type a query (AND semantics). Type 'exit' to quit.")

    while True:
        query = input("\nsearch> ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        results = and_search(query, doc_id_map, top_k=top_k)
        if not results:
            print("No results found.")
            continue

        for rank, item in enumerate(results, start=1):
            print(f"{rank}. {item['url']}  (doc_id={item['doc_id']}, score={item['score']})")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Boolean AND-only retrieval for milestone 2")
    parser.add_argument("--query", type=str, help="Single query to run")
    parser.add_argument("--topk", type=int, default=5, help="Top K results (default: 5)")
    parser.add_argument(
        "--milestone2",
        action="store_true",
        help="Run the 4 required milestone queries",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start text-based search interface",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional JSON output path for saving query results",
    )
    args = parser.parse_args()

    doc_id_map = build_doc_id_map_if_missing()

    if args.milestone2:
        milestone_results = run_milestone_queries(doc_id_map, top_k=args.topk)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(milestone_results, f, ensure_ascii=False, indent=2)
            print(f"Saved results to {args.output}")
        return

    if args.query:
        results = and_search(args.query, doc_id_map, top_k=args.topk)
        if not results:
            print("No results found.")
            return
        for rank, item in enumerate(results, start=1):
            print(f"{rank}. {item['url']}  (doc_id={item['doc_id']}, score={item['score']})")
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump({args.query: results}, f, ensure_ascii=False, indent=2)
            print(f"Saved results to {args.output}")
        return

    interactive_mode(doc_id_map, top_k=args.topk)


if __name__ == "__main__":
    main()
