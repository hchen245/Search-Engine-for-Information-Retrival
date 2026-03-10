import os
import json
import re
import heapq
import hashlib
import time
import warnings
from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning
from nltk.stem import PorterStemmer
from collections import defaultdict

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

"""Indexer for Assignment 3.

Pipeline:
1) Read crawled JSON documents from DEV/
2) Extract visible text + important HTML text (title/headings/bold)
3) Tokenize + stem
4) Build weighted term frequencies per document
5) Flush partial inverted indexes to disk
6) Merge partial indexes into final index file
7) Save doc_id -> URL mapping for retrieval component
"""

DATA_PATH = "DEV"

MAX_TERMS_IN_MEMORY = 50000
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
SPARSE_STRIDE = 128
SIMHASH_BITS = 64
SIMHASH_HAMMING_THRESHOLD = 3
SIMHASH_BANDS = 4
PAGERANK_DAMPING = 0.85
PAGERANK_MAX_ITER = 30
PAGERANK_TOL = 1e-8
ANCHOR_BOOST = 3
BIGRAM_PREFIX = "__bg__"
BIGRAM_BOOST = 2
PROGRESS_EVERY = 1000

stemmer = PorterStemmer()


def strip_fragment(url):
    """Return URL without fragment part (#...)."""
    if not isinstance(url, str):
        return ""
    return url.split("#", 1)[0]


def normalize_url(url):
    """Normalize URL for link graph matching."""
    if not isinstance(url, str):
        return ""

    clean_url = strip_fragment(url.strip())
    if not clean_url:
        return ""

    parsed = urlparse(clean_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""

    path = parsed.path if parsed.path else "/"
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            "",
            parsed.query,
            "",
        )
    )


def extract_text_from_html(html_content, base_url):
    """Extract visible text, important text, outlinks, and anchor texts with targets."""
    soup = BeautifulSoup(html_content, 'lxml')
    # Keep important fields separately so they can be weighted higher.
    important_text = []
    outlinks = set()
    anchor_pairs = []
    
    # Title
    title = soup.find('title')
    if title:
        important_text.append(title.get_text())
    
    # Headings (h1, h2, h3)
    for heading in soup.find_all(['h1', 'h2', 'h3']):
        important_text.append(heading.get_text())
    
    # Bold text
    for bold in soup.find_all(['b', 'strong']):
        important_text.append(bold.get_text())

    # Outgoing links (absolute, normalized)
    for anchor in soup.find_all('a', href=True):
        href = anchor.get('href')
        if not href:
            continue
        absolute_url = urljoin(base_url, href)
        normalized = normalize_url(absolute_url)
        if not normalized:
            continue

        outlinks.add(normalized)
        anchor_text = anchor.get_text(separator=' ', strip=True)
        if anchor_text:
            anchor_pairs.append((normalized, anchor_text))
    
    return soup.get_text(separator=' '), ' '.join(important_text), outlinks, anchor_pairs


def tokenize(text):
    """Split text into lowercase alphanumeric tokens."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return tokens


def stem_tokens(tokens):
    """Apply Porter stemming to each token."""
    result = []
    for token in tokens:
        result.append(stemmer.stem(token))
    return result


def build_bigram_terms(tokens):
    """Generate 2-gram terms with a dedicated prefix."""
    if len(tokens) < 2:
        return []
    return [f"{BIGRAM_PREFIX}{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]


def compute_simhash(tokens, bits=SIMHASH_BITS):
    """Compute SimHash fingerprint from token frequencies."""
    if not tokens:
        return 0

    token_freq = defaultdict(int)
    for token in tokens:
        token_freq[token] += 1

    vector = [0] * bits
    for token, weight in token_freq.items():
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        token_hash = int(digest, 16)
        for bit_idx in range(bits):
            if (token_hash >> bit_idx) & 1:
                vector[bit_idx] += weight
            else:
                vector[bit_idx] -= weight

    fingerprint = 0
    for bit_idx, value in enumerate(vector):
        if value >= 0:
            fingerprint |= (1 << bit_idx)
    return fingerprint


def hamming_distance(a, b):
    """Return Hamming distance between two integer bit fingerprints."""
    return bin(a ^ b).count("1")


def is_near_duplicate(simhash_value, bucket_map, simhash_store):
    """Check if document is near duplicate using banded candidate lookup."""
    band_size = SIMHASH_BITS // SIMHASH_BANDS
    candidate_ids = set()

    for band_idx in range(SIMHASH_BANDS):
        shift = band_idx * band_size
        mask = (1 << band_size) - 1
        band_key = (band_idx, (simhash_value >> shift) & mask)
        candidate_ids.update(bucket_map.get(band_key, []))

    for candidate_doc_id in candidate_ids:
        candidate_hash = simhash_store[candidate_doc_id]
        if hamming_distance(simhash_value, candidate_hash) <= SIMHASH_HAMMING_THRESHOLD:
            return True

    return False


def register_simhash(doc_id, simhash_value, bucket_map):
    """Register one accepted document fingerprint into LSH-style buckets."""
    band_size = SIMHASH_BITS // SIMHASH_BANDS
    for band_idx in range(SIMHASH_BANDS):
        shift = band_idx * band_size
        mask = (1 << band_size) - 1
        band_key = (band_idx, (simhash_value >> shift) & mask)
        bucket_map[band_key].append(doc_id)


def compute_pagerank(adjacency, total_docs):
    """Compute PageRank scores for accepted documents."""
    if total_docs <= 0:
        return {}

    docs = list(range(1, total_docs + 1))
    rank = {doc_id: 1.0 / total_docs for doc_id in docs}

    for _ in range(PAGERANK_MAX_ITER):
        new_rank = {doc_id: (1.0 - PAGERANK_DAMPING) / total_docs for doc_id in docs}
        dangling_mass = 0.0

        for doc_id in docs:
            targets = adjacency.get(doc_id, set())
            if not targets:
                dangling_mass += rank[doc_id]
                continue

            contribution = rank[doc_id] / len(targets)
            for target_id in targets:
                new_rank[target_id] += PAGERANK_DAMPING * contribution

        dangling_share = PAGERANK_DAMPING * dangling_mass / total_docs
        for doc_id in docs:
            new_rank[doc_id] += dangling_share

        delta = sum(abs(new_rank[doc_id] - rank[doc_id]) for doc_id in docs)
        rank = new_rank
        if delta < PAGERANK_TOL:
            break

    return rank


def build_index_for_one_doc(doc_id, tokens, important_tokens, anchor_tokens, inverted_index, positional_index):
    """Build weighted TF postings and positional postings for one document."""
    term_freq = defaultdict(int)
    
    # Regular tokens count as 1
    for position, token in enumerate(tokens):
        term_freq[token] += 1
        positional_index[token].setdefault(doc_id, []).append(position)
    
    # Important tokens count as 5 (boost factor)
    for token in important_tokens:
        term_freq[token] += 5

    # Anchor tokens pointing to this page get a small boost.
    for token in anchor_tokens:
        term_freq[token] += ANCHOR_BOOST

    # Index 2-gram terms for phrase-aware retrieval.
    for bigram_term in build_bigram_terms(tokens):
        term_freq[bigram_term] += BIGRAM_BOOST
    
    for term, tf in term_freq.items():
        inverted_index[term][doc_id] = tf

    return sum(term_freq.values())


def write_partial_index(inverted_index, part_num):
    """Write one in-memory chunk of inverted index to disk."""
    os.makedirs(PARTIAL_INDEX_DIR, exist_ok=True)
    filename = os.path.join(PARTIAL_INDEX_DIR, f"partial_{part_num}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        for term in sorted(inverted_index.keys()):
            postings = inverted_index[term]
            # term: docID1 tf1 docID2 tf2 ...
            postings_str = " ".join(
                f"{doc_id} {postings[doc_id]}" 
                for doc_id in sorted(postings.keys())
            )
            f.write(f"{term}: {postings_str}\n")
    print(f"Written partial index {filename}")


def write_partial_positional_index(positional_index, part_num):
    """Write one in-memory chunk of positional index to disk."""
    os.makedirs(PARTIAL_INDEX_DIR, exist_ok=True)
    filename = os.path.join(PARTIAL_INDEX_DIR, f"partial_pos_{part_num}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        for term in sorted(positional_index.keys()):
            postings = positional_index[term]
            entries = []
            for doc_id in sorted(postings.keys()):
                positions = ",".join(str(pos) for pos in postings[doc_id])
                entries.append(f"{doc_id} {positions}")
            f.write(f"{term}: {'; '.join(entries)}\n")
    print(f"Written positional partial index {filename}")


def parse_postings_line(line):
    """Parse one postings line: 'term: docID1 tf1 docID2 tf2 ...'."""
    if ":" not in line:
        return None, None

    term, postings_str = line.strip().split(":", 1)
    parts = postings_str.strip().split()
    if len(parts) % 2 != 0:
        return term, {}

    postings = {}
    for i in range(0, len(parts), 2):
        try:
            doc_id = int(parts[i])
            tf = int(parts[i + 1])
            postings[doc_id] = tf
        except ValueError:
            continue
    return term, postings


def parse_positions_line(line):
    """Parse one positional postings line: 'term: docID p1,p2; docID p1,...'."""
    if ":" not in line:
        return None, None

    term, postings_str = line.strip().split(":", 1)
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

    return term, postings


def merge_partials_streaming():
    """Merge sorted partial indexes by streaming (low memory)."""
    os.makedirs(FINAL_INDEX_DIR, exist_ok=True)

    partial_files = [
        os.path.join(PARTIAL_INDEX_DIR, name)
        for name in sorted(os.listdir(PARTIAL_INDEX_DIR))
        if name.startswith("partial_") and name.endswith(".txt")
    ]

    if not partial_files:
        raise FileNotFoundError("No partial index files found.")

    handles = []
    heap = []

    for idx, path in enumerate(partial_files):
        handle = open(path, "r", encoding="utf-8")
        handles.append(handle)
        line = handle.readline()
        if line:
            term, postings = parse_postings_line(line)
            if term is not None:
                heapq.heappush(heap, (term, idx, postings))

    term_count = 0
    with open(FINAL_INDEX_FILE, "w", encoding="utf-8") as fout, open(
        LEXICON_PATH, "w", encoding="utf-8"
    ) as flex, open(LEXICON_SPARSE_PATH, "w", encoding="utf-8") as fsparse:
        while heap:
            current_term = heap[0][0]
            merged_postings = defaultdict(int)

            while heap and heap[0][0] == current_term:
                _, file_idx, postings = heapq.heappop(heap)
                for doc_id, tf in postings.items():
                    merged_postings[doc_id] += tf

                next_line = handles[file_idx].readline()
                if next_line:
                    next_term, next_postings = parse_postings_line(next_line)
                    if next_term is not None:
                        heapq.heappush(heap, (next_term, file_idx, next_postings))

            offset = fout.tell()
            postings_items = sorted(merged_postings.items())
            postings_str = " ".join(f"{doc_id} {tf}" for doc_id, tf in postings_items)
            fout.write(f"{current_term}: {postings_str}\n")
            lexicon_line_offset = flex.tell()
            flex.write(f"{current_term}\t{offset}\t{len(postings_items)}\n")
            if term_count % SPARSE_STRIDE == 0:
                fsparse.write(f"{current_term}\t{lexicon_line_offset}\n")
            term_count += 1

    for handle in handles:
        handle.close()

    print(f"Final index written to {FINAL_INDEX_FILE}")
    print(f"Lexicon written to {LEXICON_PATH}")
    print(f"Sparse lexicon written to {LEXICON_SPARSE_PATH}")
    return term_count


def merge_positional_partials_streaming():
    """Merge sorted positional partial indexes by streaming (low memory)."""
    os.makedirs(FINAL_INDEX_DIR, exist_ok=True)

    partial_files = [
        os.path.join(PARTIAL_INDEX_DIR, name)
        for name in sorted(os.listdir(PARTIAL_INDEX_DIR))
        if name.startswith("partial_pos_") and name.endswith(".txt")
    ]

    if not partial_files:
        raise FileNotFoundError("No positional partial index files found.")

    handles = []
    heap = []

    for idx, path in enumerate(partial_files):
        handle = open(path, "r", encoding="utf-8")
        handles.append(handle)
        line = handle.readline()
        if line:
            term, postings = parse_positions_line(line)
            if term is not None:
                heapq.heappush(heap, (term, idx, postings))

    term_count = 0
    with open(POSITION_INDEX_FILE, "w", encoding="utf-8") as fout, open(
        POSITION_LEXICON_PATH, "w", encoding="utf-8"
    ) as flex, open(POSITION_SPARSE_PATH, "w", encoding="utf-8") as fsparse:
        while heap:
            current_term = heap[0][0]
            merged_postings = defaultdict(list)

            while heap and heap[0][0] == current_term:
                _, file_idx, postings = heapq.heappop(heap)
                for doc_id, positions in postings.items():
                    merged_postings[doc_id].extend(positions)

                next_line = handles[file_idx].readline()
                if next_line:
                    next_term, next_postings = parse_positions_line(next_line)
                    if next_term is not None:
                        heapq.heappush(heap, (next_term, file_idx, next_postings))

            offset = fout.tell()
            postings_items = sorted(merged_postings.items())
            entries = []
            for doc_id, positions in postings_items:
                positions.sort()
                pos_str = ",".join(str(pos) for pos in positions)
                entries.append(f"{doc_id} {pos_str}")
            fout.write(f"{current_term}: {'; '.join(entries)}\n")

            lexicon_line_offset = flex.tell()
            flex.write(f"{current_term}\t{offset}\t{len(postings_items)}\n")
            if term_count % SPARSE_STRIDE == 0:
                fsparse.write(f"{current_term}\t{lexicon_line_offset}\n")
            term_count += 1

    for handle in handles:
        handle.close()

    print(f"Positional index written to {POSITION_INDEX_FILE}")
    print(f"Positional lexicon written to {POSITION_LEXICON_PATH}")
    print(f"Positional sparse lexicon written to {POSITION_SPARSE_PATH}")
    return term_count


if __name__ == "__main__":
    doc_id_to_url = {}
    doc_id = 1
    near_duplicate_skips = 0
    processed_docs = 0

    simhash_store = {}
    simhash_bucket_map = defaultdict(list)

    accepted_docs = []
    outlinks_by_doc_id = {}
    anchor_pairs_by_doc_id = {}

    if os.path.isdir(PARTIAL_INDEX_DIR):
        for filename in os.listdir(PARTIAL_INDEX_DIR):
            if (filename.startswith("partial_") or filename.startswith("partial_pos_")) and filename.endswith(".txt"):
                try:
                    os.remove(os.path.join(PARTIAL_INDEX_DIR, filename))
                except OSError:
                    pass

    print("Pass 1/2: dedup + link/anchor extraction...", flush=True)
    pass1_start = time.time()
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if not file.endswith(".json"):
                continue

            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            raw_url = strip_fragment(data.get("url", ""))
            base_url = raw_url if raw_url else ""
            text, _, outlinks, anchor_pairs = extract_text_from_html(data['content'], base_url)
            tokens = stem_tokens(tokenize(text))

            processed_docs += 1
            if processed_docs % PROGRESS_EVERY == 0:
                elapsed = time.time() - pass1_start
                print(
                    f"[Pass1] processed={processed_docs}, accepted={doc_id - 1}, near_dup_skipped={near_duplicate_skips}, elapsed={elapsed:.1f}s",
                    flush=True,
                )

            simhash_value = compute_simhash(tokens)
            if is_near_duplicate(simhash_value, simhash_bucket_map, simhash_store):
                near_duplicate_skips += 1
                continue

            current_doc_id = doc_id
            doc_id_to_url[current_doc_id] = raw_url
            simhash_store[current_doc_id] = simhash_value
            register_simhash(current_doc_id, simhash_value, simhash_bucket_map)

            accepted_docs.append(
                {
                    "doc_id": current_doc_id,
                    "path": path,
                    "normalized_url": normalize_url(raw_url),
                }
            )
            outlinks_by_doc_id[current_doc_id] = outlinks
            anchor_pairs_by_doc_id[current_doc_id] = anchor_pairs
            doc_id += 1

    total_docs = doc_id - 1

    url_to_doc_id = {}
    for record in accepted_docs:
        normalized_url = record["normalized_url"]
        if normalized_url and normalized_url not in url_to_doc_id:
            url_to_doc_id[normalized_url] = record["doc_id"]

    anchor_tokens_by_target_doc = defaultdict(list)
    adjacency = {current_doc_id: set() for current_doc_id in range(1, total_docs + 1)}

    for source_doc_id in range(1, total_docs + 1):
        outlinks = outlinks_by_doc_id.get(source_doc_id, set())
        linked_doc_ids = set()
        for outlink_url in outlinks:
            target_doc_id = url_to_doc_id.get(outlink_url)
            if target_doc_id is not None and target_doc_id != source_doc_id:
                linked_doc_ids.add(target_doc_id)
        adjacency[source_doc_id] = linked_doc_ids

        for target_url, anchor_text in anchor_pairs_by_doc_id.get(source_doc_id, []):
            target_doc_id = url_to_doc_id.get(target_url)
            if target_doc_id is None:
                continue
            tokens = stem_tokens(tokenize(anchor_text))
            if tokens:
                anchor_tokens_by_target_doc[target_doc_id].extend(tokens)

    print("Pass 2/2: building content + positional indexes...", flush=True)
    inverted_index = defaultdict(dict)
    positional_index = defaultdict(dict)
    doc_lengths = {}
    part_num = 1
    pass2_start = time.time()
    pass2_processed = 0

    for record in accepted_docs:
        current_doc_id = record["doc_id"]
        with open(record["path"], 'r', encoding='utf-8') as f:
            data = json.load(f)

        doc_url = strip_fragment(data.get("url", ""))
        base_url = doc_url if doc_url else ""
        text, important_text, _, _ = extract_text_from_html(data['content'], base_url)

        tokens = stem_tokens(tokenize(text))
        important_tokens = stem_tokens(tokenize(important_text))
        anchor_tokens = anchor_tokens_by_target_doc.get(current_doc_id, [])

        doc_length = build_index_for_one_doc(
            current_doc_id,
            tokens,
            important_tokens,
            anchor_tokens,
            inverted_index,
            positional_index,
        )
        doc_lengths[current_doc_id] = doc_length
        pass2_processed += 1

        if pass2_processed % PROGRESS_EVERY == 0:
            elapsed = time.time() - pass2_start
            print(
                f"[Pass2] processed={pass2_processed}/{total_docs}, in_memory_terms={len(inverted_index)}, elapsed={elapsed:.1f}s",
                flush=True,
            )

        if len(inverted_index) >= MAX_TERMS_IN_MEMORY:
            print("Writing partial indexes:", part_num)
            write_partial_index(inverted_index, part_num)
            write_partial_positional_index(positional_index, part_num)
            inverted_index.clear()
            positional_index.clear()
            part_num += 1

    if inverted_index:
        write_partial_index(inverted_index, part_num)
        write_partial_positional_index(positional_index, part_num)
        inverted_index.clear()
        positional_index.clear()

    print("Merging content partial indexes (streaming)...")
    term_count = merge_partials_streaming()
    print("Merging positional partial indexes (streaming)...")
    positional_term_count = merge_positional_partials_streaming()

    os.makedirs(FINAL_INDEX_DIR, exist_ok=True)
    with open(DOC_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(doc_id_to_url, f, ensure_ascii=False)
    print(f"Document ID to URL map written to {DOC_MAP_PATH}")

    avg_doc_len = (sum(doc_lengths.values()) / total_docs) if total_docs > 0 else 0.0
    doc_meta = {
        "total_docs": total_docs,
        "avg_doc_len": avg_doc_len,
        "doc_lengths": {str(doc_key): length for doc_key, length in doc_lengths.items()},
    }
    with open(DOC_META_PATH, "w", encoding="utf-8") as f:
        json.dump(doc_meta, f, ensure_ascii=False)
    print(f"Document metadata written to {DOC_META_PATH}")

    pagerank_scores = compute_pagerank(adjacency, total_docs)
    with open(PAGERANK_PATH, "w", encoding="utf-8") as f:
        json.dump({str(doc_key): score for doc_key, score in pagerank_scores.items()}, f, ensure_ascii=False)
    print(f"PageRank written to {PAGERANK_PATH}")

    print("Total documents:", total_docs)
    print("Processed documents:", processed_docs)
    print("Near-duplicate skipped:", near_duplicate_skips)
    print("Unique content tokens:", term_count)
    print("Unique positional tokens:", positional_term_count)

    size_kb = os.path.getsize(FINAL_INDEX_FILE) / 1024
    pos_size_kb = os.path.getsize(POSITION_INDEX_FILE) / 1024
    print("Index size (KB):", round(size_kb, 2))
    print("Positional index size (KB):", round(pos_size_kb, 2))