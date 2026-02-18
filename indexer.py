import os
import json
import re
import nltk
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from collections import defaultdict

DATA_PATH = "DEV"

MAX_TERMS_IN_MEMORY = 50000

stemmer = PorterStemmer()

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'lxml')
    # Extract important text separately
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
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return tokens


def stem_tokens(tokens):
    result = []
    for token in tokens:
        result.append(stemmer.stem(token))
    return result

def build_index_for_one_doc(doc_id, tokens, important_tokens, inverted_index):
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
    find_index = defaultdict(dict)

    partial_path = "partial_indexes"

    for filename in sorted(os.listdir(partial_path)):
        if filename.startswith("partial_"):
            filepath = os.path.join(partial_path, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    term, postings_str = line.strip().split(":", 1)
                    postings_list = postings_str.strip().split()

                    for i in range(0, len(postings_list), 2): # docID tf pairs
                        doc_id = int(postings_list[i])
                        tf = int(postings_list[i+1])
                        find_index[term][doc_id] = tf
    return find_index

def write_final_index(find_index): #to disk
    os.makedirs("final_index", exist_ok=True)

    filename = "final_index/final_index.txt"
    with open(filename, "w", encoding="utf-8") as f:
        for term in sorted(find_index.keys()):
            postings = find_index[term] # {doc_id: tf, ...}
            postings_str = " ".join(f"{doc_id} {tf}" for doc_id, tf in sorted(postings.items())) # term: docID1 tf1 docID2 tf2 ...
            f.write(f"{term}: {postings_str}\n")
    print("Final index written to disk.")

if __name__ == "__main__":
    inverted_index = defaultdict(dict) # term -> {doc_id: tf, ...}
    doc_id = 1
    part_num = 1

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                text, important_text = extract_text_from_html(data['content'])
                tokens = tokenize(text)
                tokens = stem_tokens(tokens)
                
                important_tokens = tokenize(important_text)
                important_tokens = stem_tokens(important_tokens)
                
                build_index_for_one_doc(doc_id, tokens, important_tokens, inverted_index)
                doc_id += 1

                #check if we need to write to partial index
                if len(inverted_index) >= MAX_TERMS_IN_MEMORY:
                    write_partial_index(inverted_index, part_num)
                    inverted_index.clear()
                    part_num += 1

    #write remaining index
    if inverted_index:
        write_partial_index(inverted_index, part_num)   
        inverted_index.clear()

    print("Merging partial indexes...")
    final_index = merge_partials()
    write_final_index(final_index)

    print("Total documents:", doc_id - 1)
    print("Unique tokens:", len(final_index))

    size_kb = os.path.getsize("final_index/final_index.txt") / 1024
    print("Index size (KB):", round(size_kb, 2))