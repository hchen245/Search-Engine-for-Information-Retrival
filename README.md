# Search Engine for Information Retrieval

A lightweight search engine project for UCI ICS web crawl data.

This project includes:
- **Indexer (`indexer.py`)**: builds a weighted inverted index from crawled HTML JSON files.
- **Retriever (`search.py`)**: supports Boolean retrieval (`AND`/`OR`) with TF-IDF ranking.

## Features

- Parse HTML and extract visible text with important tags boost (`title`, `h1-h3`, `b/strong`)
- Tokenization + Porter stemming
- Partial index writing for memory control
- Final merged inverted index output
- `doc_id -> URL` mapping for retrieval output
- Milestone 2 required query runner
- Interactive command-line search mode

## Project Structure

- `indexer.py` — build index files
- `search.py` — query/search interface
- `DEV/` — crawled JSON dataset
- `partial_indexes/` — intermediate postings files
- `final_index/` — final index and doc map output
- `milestone2_results.json` — sample/query output file

## Environment

- Python 3.9+
- Recommended: virtual environment (`.venv`)

Install dependencies:

```bash
pip install beautifulsoup4 lxml nltk
```

## Build the Index

Run indexing first (required before search):

```bash
python indexer.py
```

Outputs:
- `partial_indexes/partial_*.txt`
- `final_index/final_index.txt`
- `final_index/doc_id_map.json`

## Search Usage

### 1) Run Milestone 2 required queries

```bash
python search.py --milestone2
```

Optional output file:

```bash
python search.py --milestone2 --output milestone2_results.json
```

### 2) Run a single query

```bash
python search.py --query "machine learning" --topk 5 --mode and
```

### 3) Interactive mode

```bash
python search.py --interactive --mode and --topk 5
```

## Search Options

- `--query "..."` : run one query
- `--milestone2` : run the 4 required milestone queries
- `--interactive` : start CLI search loop
- `--mode {and,or}` : Boolean retrieval mode (default: `and`)
- `--topk N` : number of returned results (default: `5`)
- `--output path.json` : save results to JSON

## Milestone 2 Query Set

The built-in query set is:
1. `cristina lopes`
2. `machine learning`
3. `ACM`
4. `master of software engineering`

## Notes

- If `final_index/doc_id_map.json` is missing, `search.py` can rebuild it automatically from `DEV/`.
- For reproducible grading/demo output, keep `milestone2_results.json` in the repository.
