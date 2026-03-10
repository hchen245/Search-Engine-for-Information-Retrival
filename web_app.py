import argparse
from flask import Flask, request, render_template_string

from search import build_doc_id_map_if_missing, and_search


app = Flask(__name__)
DOC_ID_MAP = build_doc_id_map_if_missing()


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IR Search Web</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 900px; margin: 24px auto; padding: 0 16px; }
    h1 { margin-bottom: 8px; }
    form { display: grid; gap: 12px; margin: 16px 0 20px; }
    input[type=text] { padding: 10px; font-size: 16px; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
    select, input[type=number] { padding: 6px; }
    button { padding: 8px 14px; cursor: pointer; }
    .meta { color: #555; font-size: 14px; margin-bottom: 12px; }
    .result { border: 1px solid #ddd; border-radius: 8px; padding: 10px 12px; margin-bottom: 10px; }
    .url { word-break: break-all; }
  </style>
</head>
<body>
  <h1>Information Retrieval Search</h1>
  <div class="meta">Web interface powered by your local index.</div>

  <form method="get" action="/">
    <input type="text" name="q" placeholder="Enter query..." value="{{ query }}" required>
    <div class="row">
      <label>Mode
        <select name="mode">
          <option value="hybrid" {% if mode == 'hybrid' %}selected{% endif %}>hybrid</option>
          <option value="strict" {% if mode == 'strict' %}selected{% endif %}>strict</option>
        </select>
      </label>
      <label>Ranking
        <select name="ranking">
          <option value="bm25" {% if ranking == 'bm25' %}selected{% endif %}>bm25</option>
          <option value="tfidf" {% if ranking == 'tfidf' %}selected{% endif %}>tfidf</option>
        </select>
      </label>
      <label>Top K
        <input type="number" name="topk" min="1" max="50" value="{{ topk }}">
      </label>
      <label>PR Weight
        <input type="number" name="pr_weight" min="0" max="1" step="0.05" value="{{ pr_weight }}">
      </label>
      <label>
        <input type="checkbox" name="use_pr" value="1" {% if use_pr %}checked{% endif %}> use PageRank
      </label>
      <button type="submit">Search</button>
    </div>
  </form>

  {% if searched %}
    <div class="meta">{{ results|length }} result(s)</div>
    {% if results %}
      {% for item in results %}
        <div class="result">
          <div><strong>#{{ loop.index }}</strong> score={{ item['score'] }} matched={{ item['matched_terms'] }} time_ms={{ item['elapsed_ms'] }}</div>
          <div class="url"><a href="{{ item['url'] }}" target="_blank">{{ item['url'] }}</a></div>
        </div>
      {% endfor %}
    {% else %}
      <div>No results found.</div>
    {% endif %}
  {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    query = request.args.get("q", "").strip()
    mode = request.args.get("mode", "hybrid")
    ranking = request.args.get("ranking", "bm25")

    try:
        topk = int(request.args.get("topk", 5))
    except ValueError:
        topk = 5
    topk = max(1, min(topk, 50))

    try:
        pr_weight = float(request.args.get("pr_weight", 0.2))
    except ValueError:
        pr_weight = 0.2
    pr_weight = max(0.0, min(pr_weight, 1.0))

    use_pr = request.args.get("use_pr") == "1"

    results = []
    searched = False
    if query:
        searched = True
        results = and_search(
            query,
            DOC_ID_MAP,
            top_k=topk,
            mode=mode if mode in {"strict", "hybrid"} else "hybrid",
            ranking=ranking if ranking in {"tfidf", "bm25"} else "bm25",
            use_pagerank=use_pr,
            pr_weight=pr_weight,
        )

    return render_template_string(
        HTML_TEMPLATE,
        query=query,
        mode=mode,
        ranking=ranking,
        topk=topk,
        pr_weight=pr_weight,
        use_pr=use_pr,
        results=results,
        searched=searched,
    )


def main():
    parser = argparse.ArgumentParser(description="Run local web search interface")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
