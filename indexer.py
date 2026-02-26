import os
import json
import re
import nltk
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from collections import defaultdict

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

stemmer = PorterStemmer()


def extract_text_from_html(html_content):
    """Extract full visible text and boosted-important text from HTML."""
    soup = BeautifulSoup(html_content, 'lxml')
    # Keep important fields separately so they can be weighted higher.
    important_text = []
    
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
    
    return soup.get_text(separator=' '), ' '.join(important_text)


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


def build_index_for_one_doc(doc_id, tokens, important_tokens, inverted_index):
    """Build weighted term frequencies for a single document.

    Normal tokens contribute +1, important tokens contribute +5.
    """
    term_freq = defaultdict(int)
    
    # Regular tokens count as 1
    for token in tokens:
        term_freq[token] += 1
    
    # Important tokens count as 5 (boost factor)
    for token in important_tokens:
        term_freq[token] += 5
    
    for term, tf in term_freq.items():
        inverted_index[term][doc_id] = tf


def write_partial_index(inverted_index, part_num):
    """Write one in-memory chunk of inverted index to disk."""
    os.makedirs("partial_indexes", exist_ok=True)
    filename = f"partial_indexes/partial_{part_num}.txt"
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


def merge_partials():
    """Merge all partial index files into one final in-memory index."""
    find_index = defaultdict(dict)
    partial_path = "partial_indexes"
    print("Merging partial indexes...")

    for filename in sorted(os.listdir(partial_path)):
        print("Merging:", filename)  # DEBUG
        if filename.startswith("partial_"):
            filepath = os.path.join(partial_path, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if ":" not in line:
                        continue

                    term, postings_str = line.strip().split(":", 1)
                    postings_list = postings_str.strip().split()

                    if len(postings_list) % 2 != 0:
                        continue

                    # Parse postings as docID/tf pairs.
                    for i in range(0, len(postings_list), 2):
                        try:
                            doc_id = int(postings_list[i])
                            tf = int(postings_list[i+1])
                            find_index[term][doc_id] = tf
                        except:
                            continue
    return find_index


def write_final_index(find_index):
    """Persist merged inverted index to final_index/final_index.txt."""
    os.makedirs("final_index", exist_ok=True)

    filename = "final_index/final_index.txt"
    with open(filename, "w", encoding="utf-8") as f:
        for term in sorted(find_index.keys()):
            postings = find_index[term] # {doc_id: tf, ...}
            postings_str = " ".join(f"{doc_id} {tf}" for doc_id, tf in sorted(postings.items())) # term: docID1 tf1 docID2 tf2 ...
            f.write(f"{term}: {postings_str}\n")
    print("Final index written to disk.")


if __name__ == "__main__":
    # In-memory structure: term -> {doc_id: tf}
    inverted_index = defaultdict(dict)
    # Needed by retrieval to convert result doc IDs back to URLs.
    doc_id_to_url = {}
    doc_id = 1
    part_num = 1

    print("Starting document processing...")  # DEBUG
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            print("Processing:", file)  # DEBUG
            if file.endswith(".json"):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                doc_id_to_url[doc_id] = data.get("url", "")
                    
                text, important_text = extract_text_from_html(data['content'])
                tokens = tokenize(text)
                tokens = stem_tokens(tokens)
                
                important_tokens = tokenize(important_text)
                important_tokens = stem_tokens(important_tokens)
                
                build_index_for_one_doc(doc_id, tokens, important_tokens, inverted_index)
                doc_id += 1

                # Flush to disk when term dictionary gets too large.
                if len(inverted_index) >= MAX_TERMS_IN_MEMORY:
                    print("Writing partial index:", part_num)  # DEBUG
                    write_partial_index(inverted_index, part_num)
                    inverted_index.clear()
                    part_num += 1

    # Write the remaining in-memory terms.
    if inverted_index:
        write_partial_index(inverted_index, part_num)   
        inverted_index.clear()

    print("Merging partial indexes...")
    final_index = merge_partials()
    write_final_index(final_index)

    # Save doc_id -> URL mapping used by search.py.
    os.makedirs("final_index", exist_ok=True)
    with open("final_index/doc_id_map.json", "w", encoding="utf-8") as f:
        json.dump(doc_id_to_url, f, ensure_ascii=False)
    print("Document ID to URL map written to final_index/doc_id_map.json")

    print("Total documents:", doc_id - 1)
    print("Unique tokens:", len(final_index))

    size_kb = os.path.getsize("final_index/final_index.txt") / 1024
    print("Index size (KB):", round(size_kb, 2))