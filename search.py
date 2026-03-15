import argparse
import bisect
import json
import math
import os
import re
import time
from collections import defaultdict
from functools import lru_cache

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
POSITION_INDEX_FILE = os.path.join(FINAL_INDEX_DIR, "positions_index.txt")
DOC_MAP_PATH = os.path.join(FINAL_INDEX_DIR, "doc_id_map.json")
LEXICON_PATH = os.path.join(FINAL_INDEX_DIR, "lexicon.tsv")
LEXICON_SPARSE_PATH = os.path.join(FINAL_INDEX_DIR, "lexicon_sparse.tsv")
POSITION_LEXICON_PATH = os.path.join(FINAL_INDEX_DIR, "positions_lexicon.tsv")
POSITION_SPARSE_PATH = os.path.join(FINAL_INDEX_DIR, "positions_lexicon_sparse.tsv")
DOC_META_PATH = os.path.join(FINAL_INDEX_DIR, "doc_meta.json")
PAGERANK_PATH = os.path.join(FINAL_INDEX_DIR, "pagerank.json")

MILESTONE_QUERIES = [
    "cristina lopes",
    "machine learning",
    "ACM",
    "master of software engineering",
]

MILESTONE3_BENCHMARK_QUERIES = [
    "information retrieval project",
    "graduate student forms",
    "machine learning systems",
    "data science seminar",
    "software engineering requirements",
    "database systems lab",
    "cybersecurity club meetings",
    "computer vision deep learning",
    "natural language processing course",
    "distributed systems research",
    "human computer interaction",
    "bioinformatics phd admissions",
    "cristina lopes",
    "machine learning",
    "ACM",
    "master of software engineering",
    "ics helpdesk",
    "uci informatics",
    "graduate admissions",
    "faculty directory",
    "undergrad forms",
    "tutoring center",
    "honors program",
    "research labs",
]

stemmer = PorterStemmer()
lexicon_cache = None
doc_meta_cache = None
sparse_terms = None
sparse_offsets = None
position_sparse_terms = None
position_sparse_offsets = None
pagerank_cache = None
BIGRAM_PREFIX = "__bg__"


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


def build_bigram_terms(tokens):
    """Build 2-gram term keys used by indexer/search."""
    if len(tokens) < 2:
        return []
    return [f"{BIGRAM_PREFIX}{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]


def build_doc_id_map_if_missing():
    """Load doc_id -> URL map from disk, or build it from DEV if missing.

    The mapping is needed because postings store doc IDs while report output
    requires URLs.
    """
    # Search results are stored as doc IDs, but demos/reports need URLs.
    # 检索结果内部用 doc_id，展示给用户和 TA 时需要映射回 URL。
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


def load_sparse_lexicon():
    """Load sparse checkpoints: term -> byte offset in lexicon.tsv."""
    if not os.path.exists(LEXICON_SPARSE_PATH):
        raise FileNotFoundError(
            "Missing final_index/lexicon_sparse.tsv. Re-run `python indexer.py` to generate it."
        )

    terms = [] #anchor terms for binary search in lexicon.tsv
    offsets = [] #corresponding byte offsets in lexicon.tsv
    with open(LEXICON_SPARSE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            term, offset = parts
            try:
                terms.append(term)
                offsets.append(int(offset))
            except ValueError:
                continue
    return terms, offsets


def load_sparse_position_lexicon():
    """Load sparse checkpoints for positional lexicon."""
    if not os.path.exists(POSITION_SPARSE_PATH):
        raise FileNotFoundError(
            "Missing final_index/positions_lexicon_sparse.tsv. Re-run `python indexer.py` to generate it."
        )

    terms = []
    offsets = []
    with open(POSITION_SPARSE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            term, offset = parts
            try:
                terms.append(term)
                offsets.append(int(offset))
            except ValueError:
                continue
    return terms, offsets


# This is a lexicon lookpu model that combines sparse skip pointers + local sequential scanning + LRU caching
# it replace full table sacn with short scan after binary search positioning, reducing query I/O
@lru_cache(maxsize=50000)
def find_lexicon_entry(term):
    """Find one term in lexicon.tsv using sparse checkpoints and local scan."""
    global sparse_terms
    global sparse_offsets

    if not os.path.exists(LEXICON_PATH):
        raise FileNotFoundError(
            "Missing final_index/lexicon.tsv. Re-run `python indexer.py` to generate it."
        )

    if sparse_terms is None or sparse_offsets is None:
        sparse_terms, sparse_offsets = load_sparse_lexicon()

    start_offset = 0
    if sparse_terms:
        idx = bisect.bisect_right(sparse_terms, term) - 1
        if idx >= 0:
            start_offset = sparse_offsets[idx]

    with open(LEXICON_PATH, "r", encoding="utf-8") as f:
        f.seek(start_offset)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue

            current_term, posting_offset, df = parts
            if current_term == term:
                try:
                    return int(posting_offset), int(df)
                except ValueError:
                    return None
            if current_term > term:
                return None

    return None


@lru_cache(maxsize=50000)
def find_position_lexicon_entry(term):
    """Find one term in positions_lexicon.tsv via sparse checkpoints."""
    global position_sparse_terms
    global position_sparse_offsets

    if not os.path.exists(POSITION_LEXICON_PATH):
        raise FileNotFoundError(
            "Missing final_index/positions_lexicon.tsv. Re-run `python indexer.py` to generate it."
        )

    if position_sparse_terms is None or position_sparse_offsets is None:
        position_sparse_terms, position_sparse_offsets = load_sparse_position_lexicon()

    start_offset = 0
    if position_sparse_terms:
        idx = bisect.bisect_right(position_sparse_terms, term) - 1
        if idx >= 0:
            start_offset = position_sparse_offsets[idx]

    with open(POSITION_LEXICON_PATH, "r", encoding="utf-8") as f:
        f.seek(start_offset)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue

            current_term, posting_offset, df = parts
            if current_term == term:
                try:
                    return int(posting_offset), int(df)
                except ValueError:
                    return None
            if current_term > term:
                return None

    return None


def load_doc_meta(total_docs):
    """Load document length metadata for BM25 scoring."""
    if not os.path.exists(DOC_META_PATH):
        return {
            "avg_doc_len": 1.0,
            "doc_lengths": {},
            "total_docs": total_docs,
        }

    with open(DOC_META_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_lengths = data.get("doc_lengths", {})
    doc_lengths = {}
    for doc_id, doc_len in raw_lengths.items():
        try:
            doc_lengths[int(doc_id)] = int(doc_len)
        except (TypeError, ValueError):
            continue

    avg_doc_len = data.get("avg_doc_len", 1.0)
    try:
        avg_doc_len = float(avg_doc_len)
    except (TypeError, ValueError):
        avg_doc_len = 1.0

    if avg_doc_len <= 0:
        avg_doc_len = 1.0

    return {
        "avg_doc_len": avg_doc_len,
        "doc_lengths": doc_lengths,
        "total_docs": int(data.get("total_docs", total_docs)),
    }


def load_pagerank_scores():
    """Load PageRank scores as doc_id -> float."""
    if not os.path.exists(PAGERANK_PATH):
        return {}

    with open(PAGERANK_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    scores = {}
    for doc_id, score in raw.items():
        try:
            scores[int(doc_id)] = float(score)
        except (TypeError, ValueError):
            continue
    return scores


def blend_scores_with_pagerank(scored, pagerank_scores, pr_weight):
    """Blend normalized content score with normalized PageRank."""
    if not scored:
        return []

    # Keep PageRank as a light authority signal instead of letting it dominate.
    # 把 PageRank 作为轻量级“权威度”信号，而不是压过文本相关性。
    safe_weight = max(0.0, min(1.0, pr_weight))
    if safe_weight <= 0 or not pagerank_scores:
        return scored

    max_base = max(score for _, score in scored)
    if max_base <= 0:
        max_base = 1.0

    max_pr = max(pagerank_scores.values()) if pagerank_scores else 0.0
    if max_pr <= 0:
        max_pr = 1.0

    # Normalize both sides before blending so BM25/TF-IDF scale differences do not matter.
    # 先归一化内容分数和 PageRank，避免不同分数尺度直接相加失真。
    blended = []
    for doc_id, base_score in scored:
        base_norm = base_score / max_base
        pr_norm = pagerank_scores.get(doc_id, 0.0) / max_pr
        combined_score = (1.0 - safe_weight) * base_norm + safe_weight * pr_norm
        blended.append((doc_id, combined_score))

    blended.sort(key=lambda item: (-item[1], item[0]))
    return blended


@lru_cache(maxsize=20000)
def load_term_postings_by_seek(term, byte_offset):
    """Read one postings list using byte offset from lexicon."""
    # Direct seek avoids scanning the entire inverted index for every query.
    # 通过字节偏移直接定位 postings，避免每次查询都顺序扫描整份倒排索引。
    with open(FINAL_INDEX_FILE, "r", encoding="utf-8") as f:
        f.seek(byte_offset)
        line = f.readline()
    if not line:
        return {}

    parsed_term, postings = parse_postings_line(line)
    if parsed_term != term:
        return {}
    return postings


@lru_cache(maxsize=20000)
def load_term_positions_by_seek(term, byte_offset):
    """Read one positional postings list using byte offset from positions lexicon."""
    with open(POSITION_INDEX_FILE, "r", encoding="utf-8") as f:
        f.seek(byte_offset)
        line = f.readline()
    if not line:
        return {}

    parsed_term, postings_str = line.strip().split(":", 1)
    if parsed_term != term:
        return {}

    postings = {}
    entries = [entry.strip() for entry in postings_str.split(";") if entry.strip()]
    for entry in entries:
        if " " not in entry:
            continue
        doc_id_str, pos_str = entry.split(" ", 1)
        try:
            doc_id = int(doc_id_str)
        except ValueError:
            continue

        positions = []
        for raw_pos in pos_str.split(","):
            raw_pos = raw_pos.strip()
            if not raw_pos:
                continue
            try:
                positions.append(int(raw_pos))
            except ValueError:
                continue
        if positions:
            postings[doc_id] = positions

    return postings


def load_query_postings(query_terms):
    """Load postings only for query terms using lexicon offsets."""
    if not os.path.exists(FINAL_INDEX_FILE):
        raise FileNotFoundError(
            "No final index found. Run `python indexer.py` to generate final_index/final_index.txt."
        )

    postings_by_term = {}
    for term in query_terms:
        lex_entry = find_lexicon_entry(term)
        if not lex_entry:
            postings_by_term[term] = {}
            continue

        byte_offset, _ = lex_entry
        postings_by_term[term] = load_term_postings_by_seek(term, byte_offset)

    return postings_by_term


def load_query_positions(query_terms):
    """Load positional postings only for query terms."""
    if not os.path.exists(POSITION_INDEX_FILE):
        return {}

    positions_by_term = {}
    for term in query_terms:
        lex_entry = find_position_lexicon_entry(term)
        if not lex_entry:
            positions_by_term[term] = {}
            continue

        byte_offset, _ = lex_entry
        positions_by_term[term] = load_term_positions_by_seek(term, byte_offset)
    return positions_by_term


def compute_min_cover_window(term_positions_lists):
    """Compute min window size covering one position from each term list."""
    if len(term_positions_lists) < 2:
        return None

    pointers = [0] * len(term_positions_lists)
    best = None

    # Sliding multiple pointers finds the tightest span containing all query terms.
    # 多指针滑动用于找到覆盖所有查询词的最小窗口。
    while True:
        current_positions = []
        for idx, positions in enumerate(term_positions_lists):
            pointer = pointers[idx]
            if pointer >= len(positions):
                return best
            current_positions.append((positions[pointer], idx))

        min_pos, min_idx = min(current_positions)
        max_pos, _ = max(current_positions)
        window = max_pos - min_pos + 1
        if best is None or window < best:
            best = window

        pointers[min_idx] += 1


def apply_proximity_boost(scored, unique_terms, positions_by_term):
    """Boost documents where query terms appear close together."""
    if not scored or len(unique_terms) < 2 or not positions_by_term:
        return scored

    boosted = []
    for doc_id, base_score in scored:
        lists = []
        for term in unique_terms:
            positions = positions_by_term.get(term, {}).get(doc_id, [])
            if positions:
                lists.append(positions)

        if len(lists) < 2:
            boosted.append((doc_id, base_score))
            continue

        # Smaller windows usually indicate phrase-like relevance.
        # 更小的窗口通常意味着这些词在文档中更像一个短语或紧密主题。
        min_window = compute_min_cover_window(lists)
        if min_window is None:
            boosted.append((doc_id, base_score))
            continue

        proximity_gain = 1.0 + (1.5 / max(1, min_window))
        boosted.append((doc_id, base_score * proximity_gain))

    boosted.sort(key=lambda item: (-item[1], item[0]))
    return boosted


def apply_bigram_boost(scored, query_bigram_terms, bigram_postings_by_term):
    """Boost docs that match indexed query bigrams."""
    if not scored or not query_bigram_terms:
        return scored

    boosted = []
    # Bigram matches reward local phrase consistency beyond unigram overlap.
    # 命中 2-gram 可以奖励局部短语一致性，而不仅仅是单词重合。
    total_bigrams = len(query_bigram_terms)
    for doc_id, base_score in scored:
        matched = 0
        for bigram_term in query_bigram_terms:
            if doc_id in bigram_postings_by_term.get(bigram_term, {}):
                matched += 1

        if matched <= 0:
            boosted.append((doc_id, base_score))
            continue

        ratio = matched / total_bigrams
        boosted.append((doc_id, base_score * (1.0 + 0.35 * ratio)))

    boosted.sort(key=lambda item: (-item[1], item[0]))
    return boosted


def collect_candidate_docs(postings_by_term, terms, top_k, mode="hybrid"):
    """Collect candidate docs using strict AND first, then soft fallback."""
    doc_sets_all = [set(postings_by_term.get(term, {}).keys()) for term in terms]
    strict_candidates = set.intersection(*doc_sets_all) if doc_sets_all else set()

    match_counts = defaultdict(int)
    available_terms = [term for term in terms if postings_by_term.get(term)]
    for term in available_terms:
        for doc_id in postings_by_term[term]:
            match_counts[doc_id] += 1

    # strict = pure AND; hybrid = AND first, then progressively relax overlap.
    # strict 是纯 AND；hybrid 先做 AND，再逐步降低最少匹配词数。
    if mode == "strict":
        return strict_candidates, match_counts

    if strict_candidates:
        return strict_candidates, match_counts

    if not available_terms:
        return set(), match_counts

    # Start from a relatively high overlap threshold, not plain OR.
    # 从较高重合度开始放宽，而不是一开始就退化成 OR。
    max_term_matches = len(available_terms)
    initial_min_match = max(1, math.ceil(len(terms) * 0.6))
    initial_min_match = min(initial_min_match, max_term_matches)

    for min_match in range(initial_min_match, 0, -1):
        candidates = {doc_id for doc_id, matched in match_counts.items() if matched >= min_match}
        if len(candidates) >= top_k or min_match == 1:
            return candidates, match_counts

    return set(), match_counts


def score_candidates_bm25(candidates, terms, postings_by_term, doc_meta, match_counts):
    """Rank candidates with BM25 plus coordination factor."""
    if not candidates:
        return []

    total_docs = doc_meta["total_docs"]
    avg_doc_len = doc_meta["avg_doc_len"]
    doc_lengths = doc_meta["doc_lengths"]

    # Standard BM25 parameters: k1 controls TF saturation, b controls length normalization.
    # BM25 常用参数：k1 控制词频饱和，b 控制文档长度归一化强度。
    k1 = 1.2
    b = 0.75
    unique_terms = list(dict.fromkeys(terms))

    idf = {}
    for term in unique_terms:
        df = len(postings_by_term.get(term, {}))
        if df <= 0:
            idf[term] = 0.0
            continue
        idf[term] = math.log(1 + ((total_docs - df + 0.5) / (df + 0.5)))

    scored = []
    for doc_id in candidates:
        score = 0.0
        doc_len = max(1, doc_lengths.get(doc_id, int(avg_doc_len)))

        for term in unique_terms:
            tf = postings_by_term.get(term, {}).get(doc_id, 0)
            if tf <= 0:
                continue

            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
            score += idf[term] * (numerator / denominator)

        # Coordination favors documents matching more distinct query terms.
        # coordination 会偏好命中更多不同查询词的文档。
        coordination = match_counts.get(doc_id, 0) / len(unique_terms)
        score *= (0.7 + 0.3 * coordination)
        scored.append((doc_id, score))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


def score_candidates_tfidf(candidates, terms, postings_by_term, total_docs, match_counts):
    """Rank candidates with log-TF * IDF plus coordination factor."""
    if not candidates:
        return []

    # TF-IDF is kept mainly as a baseline for comparison against BM25.
    # TF-IDF 主要作为和 BM25 对比的 baseline 保留。
    unique_terms = list(dict.fromkeys(terms))
    idf = {}
    for term in unique_terms:
        df = len(postings_by_term.get(term, {}))
        idf[term] = math.log((total_docs + 1) / (df + 1)) + 1.0

    scored = []
    for doc_id in candidates:
        score = 0.0
        for term in unique_terms:
            tf = postings_by_term.get(term, {}).get(doc_id, 0)
            if tf > 0:
                score += (1 + math.log(tf)) * idf[term]

        coordination = match_counts.get(doc_id, 0) / len(unique_terms)
        score *= (0.7 + 0.3 * coordination)
        scored.append((doc_id, score))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


def and_search(query, doc_id_map, top_k=5, mode="hybrid", ranking="bm25", use_pagerank=True, pr_weight=0.2):
    """Run retrieval and return top_k ranked URLs.

    Modes:
    - strict: pure AND
    - hybrid: strict AND first, then soft fallback
    """
    global doc_meta_cache
    global pagerank_cache

    # Query latency is measured end-to-end for benchmark reporting.
    # 查询延迟按端到端方式计时，用于 benchmark 统计。
    start = time.perf_counter()
    terms = normalize_query(query)
    if not terms:
        return []

    unique_terms = list(dict.fromkeys(terms))
    query_bigram_terms = build_bigram_terms(terms)
    # Only query-time-needed postings are loaded; the full index stays on disk.
    # 只加载当前查询需要的 postings，完整索引始终留在磁盘上。
    postings_by_term = load_query_postings(terms)
    bigram_postings_by_term = load_query_postings(query_bigram_terms)
    positions_by_term = load_query_positions(unique_terms)

    candidate_docs, match_counts = collect_candidate_docs(
        postings_by_term,
        unique_terms,
        top_k=top_k,
        mode=mode,
    )

    if not candidate_docs:
        return []

    if doc_meta_cache is None:
        doc_meta_cache = load_doc_meta(total_docs=len(doc_id_map))
        if doc_meta_cache.get("total_docs", 0) <= 0:
            doc_meta_cache["total_docs"] = len(doc_id_map)

    # Content ranking happens first, then proximity/bigram/PageRank refinements are layered on top.
    # 先做内容相关性排序，再叠加 proximity、bigram、PageRank 等增强信号。
    if ranking == "tfidf":
        scored = score_candidates_tfidf(
            candidate_docs,
            unique_terms,
            postings_by_term,
            total_docs=doc_meta_cache.get("total_docs", len(doc_id_map)),
            match_counts=match_counts,
        )
    else:
        scored = score_candidates_bm25(
            candidate_docs,
            unique_terms,
            postings_by_term,
            doc_meta_cache,
            match_counts,
        )

    scored = apply_proximity_boost(scored, unique_terms, positions_by_term)
    scored = apply_bigram_boost(scored, query_bigram_terms, bigram_postings_by_term)

    if use_pagerank:
        if pagerank_cache is None:
            pagerank_cache = load_pagerank_scores()
        scored = blend_scores_with_pagerank(scored, pagerank_cache, pr_weight)

    # Deduplicate URLs in case multiple doc IDs collapse to the same canonical URL.
    # 如果多个 doc_id 最终对应同一规范化 URL，这里只展示一次。
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
                "matched_terms": match_counts.get(doc_id, 0),
            }
        )
        if len(results) >= top_k:
            break

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    for item in results:
        item["elapsed_ms"] = round(elapsed_ms, 3)

    return results


def run_milestone_queries(
    doc_id_map,
    top_k=5,
    mode="hybrid",
    ranking="bm25",
    use_pagerank=True,
    pr_weight=0.2,
):
    """Execute the 4 required milestone queries and print top results."""
    all_results = {}
    for i, query in enumerate(MILESTONE_QUERIES, start=1):
        print("=" * 80)
        print(f"Query {i}: {query}")
        results = and_search(
            query,
            doc_id_map,
            top_k=top_k,
            mode=mode,
            ranking=ranking,
            use_pagerank=use_pagerank,
            pr_weight=pr_weight,
        )
        all_results[query] = results

        if not results:
            print("No results found.")
            continue

        for rank, item in enumerate(results, start=1):
            print(
                f"{rank}. {item['url']}  "
                f"(doc_id={item['doc_id']}, score={item['score']}, matched={item['matched_terms']}, time_ms={item['elapsed_ms']})"
            )

    return all_results


def run_milestone3_benchmark(
    doc_id_map,
    top_k=5,
    mode="hybrid",
    ranking="bm25",
    use_pagerank=True,
    pr_weight=0.2,
    output_path=None,
):
    """Run 24 benchmark queries and print latency summary."""
    benchmark_results = {}
    latencies = []

    # The benchmark mixes hard and easy queries to test both effectiveness and efficiency.
    # benchmark 同时包含难查询和易查询，用来观察效果与效率两方面表现。
    for i, query in enumerate(MILESTONE3_BENCHMARK_QUERIES, start=1):
        results = and_search(
            query,
            doc_id_map,
            top_k=top_k,
            mode=mode,
            ranking=ranking,
            use_pagerank=use_pagerank,
            pr_weight=pr_weight,
        )

        elapsed_ms = 0.0
        if results:
            elapsed_ms = results[0].get("elapsed_ms", 0.0)

        benchmark_results[query] = {
            "elapsed_ms": elapsed_ms,
            "result_count": len(results),
            "results": results,
        }
        latencies.append(elapsed_ms)
        print(f"[{i:02d}] {query} -> {round(elapsed_ms, 3)} ms, {len(results)} results")

    if latencies:
        # Report simple latency summary statistics for milestone write-up.
        # 输出基础延迟统计，便于直接写入 milestone report。
        sorted_latencies = sorted(latencies)
        p90_index = max(0, int(len(sorted_latencies) * 0.9) - 1)
        avg_latency = sum(latencies) / len(latencies)
        median_latency = sorted_latencies[len(sorted_latencies) // 2]
        max_latency = max(latencies)
        over_300 = sum(1 for t in latencies if t > 300)

        print("=" * 80)
        print("Milestone 3 benchmark summary")
        print(f"queries={len(latencies)}")
        print(f"avg_ms={round(avg_latency, 3)}")
        print(f"median_ms={round(median_latency, 3)}")
        print(f"p90_ms={round(sorted_latencies[p90_index], 3)}")
        print(f"max_ms={round(max_latency, 3)}")
        print(f"over_300ms={over_300}")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
        print(f"Saved results to {output_path}")

    return benchmark_results


def interactive_mode(doc_id_map, top_k=5, mode="hybrid", ranking="bm25", use_pagerank=True, pr_weight=0.2):
    """Start simple CLI loop for manual query testing."""
    # Useful during demos when we want to try ad-hoc queries quickly.
    # 适合 demo 时快速手动试查询。
    print(
        "Search interface started. "
        f"Mode={mode}, Ranking={ranking}. Type a query. Type 'exit' to quit."
    )

    while True:
        query = input("\nsearch> ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        results = and_search(
            query,
            doc_id_map,
            top_k=top_k,
            mode=mode,
            ranking=ranking,
            use_pagerank=use_pagerank,
            pr_weight=pr_weight,
        )
        if not results:
            print("No results found.")
            continue

        for rank, item in enumerate(results, start=1):
            print(
                f"{rank}. {item['url']}  "
                f"(doc_id={item['doc_id']}, score={item['score']}, matched={item['matched_terms']}, time_ms={item['elapsed_ms']})"
            )


def main():
    """CLI entry point."""
    # CLI keeps experiments reproducible: same query, same mode, same ranking options.
    # 命令行参数让实验可复现：相同 query、mode、ranking 都能稳定复跑。
    parser = argparse.ArgumentParser(description="Milestone 3 search (strict AND + hybrid soft fallback)")
    parser.add_argument("--query", type=str, help="Single query to run")
    parser.add_argument("--topk", type=int, default=5, help="Top K results (default: 5)")
    parser.add_argument(
        "--milestone2",
        action="store_true",
        help="Run the 4 required milestone queries",
    )
    parser.add_argument(
        "--benchmark24",
        action="store_true",
        help="Run built-in 24-query benchmark and print latency summary",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start text-based search interface",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["strict", "hybrid"],
        default="hybrid",
        help="Retrieval mode: strict AND or hybrid soft fallback (default: hybrid)",
    )
    parser.add_argument(
        "--ranking",
        type=str,
        choices=["tfidf", "bm25"],
        default="bm25",
        help="Ranking formula to use (default: bm25)",
    )
    parser.add_argument(
        "--no-pagerank",
        action="store_true",
        help="Disable PageRank blending in final ranking",
    )
    parser.add_argument(
        "--pr-weight",
        type=float,
        default=0.2,
        help="PageRank blend weight in [0,1] (default: 0.2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional JSON output path for saving query results",
    )
    args = parser.parse_args()

    doc_id_map = build_doc_id_map_if_missing()

    if args.milestone2:
        milestone_results = run_milestone_queries(
            doc_id_map,
            top_k=args.topk,
            mode=args.mode,
            ranking=args.ranking,
            use_pagerank=not args.no_pagerank,
            pr_weight=args.pr_weight,
        )
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(milestone_results, f, ensure_ascii=False, indent=2)
            print(f"Saved results to {args.output}")
        return

    if args.benchmark24:
        run_milestone3_benchmark(
            doc_id_map,
            top_k=args.topk,
            mode=args.mode,
            ranking=args.ranking,
            use_pagerank=not args.no_pagerank,
            pr_weight=args.pr_weight,
            output_path=args.output,
        )
        return

    if args.query:
        results = and_search(
            args.query,
            doc_id_map,
            top_k=args.topk,
            mode=args.mode,
            ranking=args.ranking,
            use_pagerank=not args.no_pagerank,
            pr_weight=args.pr_weight,
        )
        if not results:
            print("No results found.")
            return
        for rank, item in enumerate(results, start=1):
            print(
                f"{rank}. {item['url']}  "
                f"(doc_id={item['doc_id']}, score={item['score']}, matched={item['matched_terms']}, time_ms={item['elapsed_ms']})"
            )
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump({args.query: results}, f, ensure_ascii=False, indent=2)
            print(f"Saved results to {args.output}")
        return

    interactive_mode(
        doc_id_map,
        top_k=args.topk,
        mode=args.mode,
        ranking=args.ranking,
        use_pagerank=not args.no_pagerank,
        pr_weight=args.pr_weight,
    )


if __name__ == "__main__":
    main()
