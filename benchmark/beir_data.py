"""Download and load BEIR benchmark datasets.

Fetches the official BEIR zips (corpus.jsonl / queries.jsonl / qrels/*.tsv)
directly, so neither the `beir` package nor HuggingFace `datasets` is needed.
Datasets are cached under benchmark/datasets/ and only downloaded once.
"""
import csv
import json
import os
import re
import sys
import urllib.request
import zipfile

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

# Reference sizes to help pick a dataset (docs, test queries)
KNOWN_DATASETS = {
    "scifact":      ("5K docs, 300 queries", "scientific claim verification"),
    "nfcorpus":     ("3.6K docs, 323 queries", "medical/nutrition"),
    "arguana":      ("8.7K docs, 1406 queries", "argument retrieval"),
    "scidocs":      ("25K docs, 1000 queries", "citation prediction"),
    "fiqa":         ("57K docs, 648 queries", "financial QA"),
    "trec-covid":   ("171K docs, 50 queries", "COVID literature"),
    "quora":        ("523K docs, 10000 queries", "duplicate questions"),
    "nq":           ("2.68M docs, 3452 queries", "natural questions"),
    "msmarco":      ("8.8M docs, 6980 queries (dev split)", "web passages"),
    "juristcu":     ("16K docs, 150 queries", "Brazilian legal (Portuguese, from HF Hub)"),
}

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text):
    return _TAG_RE.sub(" ", text or "").strip()


def _load_juristcu(data_dir):
    """JurisTCU: Portuguese legal IR over TCU jurisprudence (CC-BY 4.0).

    Not part of BEIR — three CSVs fetched from the HuggingFace Hub.
    16,045 documents, 150 queries (50 search-log + 100 synthetic),
    graded qrels 0-3 (15 judged docs per query). Documents are indexed
    as TEMA/SUBTEMA (title) + ENUNCIADO + EXCERTO (text), HTML stripped.
    """
    from huggingface_hub import hf_hub_download

    target = os.path.join(data_dir, "juristcu")
    os.makedirs(target, exist_ok=True)
    files = {
        fname: hf_hub_download(repo_id="LeandroRibeiro/JurisTCU",
                               filename=fname, repo_type="dataset",
                               local_dir=target)
        for fname in ("doc.csv", "query.csv", "qrel.csv")
    }
    csv.field_size_limit(10_000_000)  # EXCERTO fields can be very long

    qrels = {}
    with open(files["qrel.csv"], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            score = int(float(row["SCORE"]))
            qrels.setdefault(row["QUERY_ID"], {})[row["DOC_ID"]] = score

    corpus = {}
    with open(files["doc.csv"], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            title = " ".join(x for x in (row.get("TEMA"), row.get("SUBTEMA")) if x)
            text = " ".join(_strip_html(row.get(c)) for c in ("ENUNCIADO", "EXCERTO"))
            corpus[row["KEY"]] = {"title": title.strip(), "text": text.strip()}

    queries = {}
    with open(files["query.csv"], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ID"] in qrels:
                queries[row["ID"]] = row["TEXT"]

    return corpus, queries, qrels


def _download(url, dest_path):
    def _progress(blocks, block_size, total_size):
        if total_size > 0:
            pct = min(blocks * block_size / total_size * 100, 100)
            sys.stdout.write(f"\rDownloading: {pct:5.1f}% of {total_size / 1e6:.1f} MB")
            sys.stdout.flush()

    print(f"Fetching {url}")
    urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
    print()


def download_and_extract(name, data_dir=DEFAULT_DATA_DIR):
    """Ensure the BEIR dataset `name` is present locally; return its folder."""
    dataset_dir = os.path.join(data_dir, name)
    if os.path.exists(os.path.join(dataset_dir, "corpus.jsonl")):
        return dataset_dir

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, f"{name}.zip")
    if not os.path.exists(zip_path):
        try:
            _download(BEIR_BASE_URL.format(name), zip_path)
        except Exception:
            if os.path.exists(zip_path):
                os.remove(zip_path)  # don't leave a partial zip behind
            raise
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    os.remove(zip_path)

    if not os.path.exists(os.path.join(dataset_dir, "corpus.jsonl")):
        raise FileNotFoundError(
            f"Extraction finished but {dataset_dir}/corpus.jsonl is missing — "
            f"unexpected archive layout for dataset '{name}'."
        )
    return dataset_dir


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_beir_dataset(name, data_dir=DEFAULT_DATA_DIR, split="test"):
    """Load a BEIR dataset.

    Returns:
        corpus: {doc_id: {"title": str, "text": str}}
        queries: {query_id: str} — only queries that appear in the qrels split
        qrels: {query_id: {doc_id: relevance_int}}
    """
    if name.lower() == "juristcu":
        return _load_juristcu(data_dir)  # no splits; `split` is ignored

    dataset_dir = download_and_extract(name, data_dir)

    qrels_path = os.path.join(dataset_dir, "qrels", f"{split}.tsv")
    if not os.path.exists(qrels_path):
        available = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(os.path.join(dataset_dir, "qrels"))
            if f.endswith(".tsv")
        )
        raise FileNotFoundError(
            f"Split '{split}' not found for '{name}'. Available splits: {available}"
        )

    qrels = {}
    with open(qrels_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip the header: query-id, corpus-id, score
        for row in reader:
            if len(row) < 3:
                continue
            qid, did, score = row[0], row[1], int(row[2])
            qrels.setdefault(qid, {})[did] = score

    corpus = {}
    for doc in _read_jsonl(os.path.join(dataset_dir, "corpus.jsonl")):
        corpus[str(doc["_id"])] = {
            "title": doc.get("title") or "",
            "text": doc.get("text") or "",
        }

    queries = {}
    for q in _read_jsonl(os.path.join(dataset_dir, "queries.jsonl")):
        qid = str(q["_id"])
        if qid in qrels:
            queries[qid] = q["text"]

    return corpus, queries, qrels
