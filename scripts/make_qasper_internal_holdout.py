from pathlib import Path
import json
import random

SEED = 42
HELDOUT_FRAC = 0.10

raw_train_path = Path("data/raw/qasper/train.json")
proc_queries_path = Path("data/processed/queries/qasper_train.jsonl")
proc_docs_path = Path("data/processed/documents/qasper_train_documents.jsonl")
proc_chunks_path = Path("data/processed/chunks/qasper_train_chunks.jsonl")

raw_out_dir = Path("data/raw/qasper")
proc_queries_dir = Path("data/processed/queries")
proc_docs_dir = Path("data/processed/documents")
proc_chunks_dir = Path("data/processed/chunks")

DOC_KEYS = {"doc_id", "paper_id", "source_doc_id", "document_id", "paper_uid", "article_id"}

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def normalize_id(x):
    if x is None:
        return None
    x = str(x)
    for prefix in ["qasper_doc_", "qasper_paper_", "doc_", "paper_"]:
        if x.startswith(prefix):
            return x[len(prefix):]
    return x

def collect_doc_ids(obj):
    found = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in DOC_KEYS and v is not None:
                    found.add(normalize_id(v))
                walk(v)
        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(obj)
    return found

with open(raw_train_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

if not isinstance(raw, dict):
    raise ValueError("Expected data/raw/qasper/train.json to be a dict keyed by raw document id")

raw_doc_ids = sorted(raw.keys())
random.Random(SEED).shuffle(raw_doc_ids)

n_heldout = max(1, int(len(raw_doc_ids) * HELDOUT_FRAC))
heldout_ids = set(raw_doc_ids[:n_heldout])
train_ids = set(raw_doc_ids[n_heldout:])

raw_train_internal = {k: v for k, v in raw.items() if k in train_ids}
raw_heldout_internal = {k: v for k, v in raw.items() if k in heldout_ids}

(raw_out_dir / "train_internal.json").write_text(
    json.dumps(raw_train_internal, ensure_ascii=False, indent=2), encoding="utf-8"
)
(raw_out_dir / "heldout_internal.json").write_text(
    json.dumps(raw_heldout_internal, ensure_ascii=False, indent=2), encoding="utf-8"
)

queries = read_jsonl(proc_queries_path)
docs = read_jsonl(proc_docs_path)
chunks = read_jsonl(proc_chunks_path)

def filter_rows(rows, allowed_ids):
    kept = []
    unmatched = 0
    for row in rows:
        ids = collect_doc_ids(row)
        if ids & allowed_ids:
            kept.append(row)
        else:
            unmatched += 1
    return kept, unmatched

train_queries, unmatched_train_queries = filter_rows(queries, train_ids)
heldout_queries, unmatched_heldout_queries = filter_rows(queries, heldout_ids)

train_docs, unmatched_train_docs = filter_rows(docs, train_ids)
heldout_docs, unmatched_heldout_docs = filter_rows(docs, heldout_ids)

train_chunks, unmatched_train_chunks = filter_rows(chunks, train_ids)
heldout_chunks, unmatched_heldout_chunks = filter_rows(chunks, heldout_ids)

write_jsonl(train_queries, proc_queries_dir / "qasper_train_internal.jsonl")
write_jsonl(heldout_queries, proc_queries_dir / "qasper_heldout_internal.jsonl")

write_jsonl(train_docs, proc_docs_dir / "qasper_train_internal_documents.jsonl")
write_jsonl(heldout_docs, proc_docs_dir / "qasper_heldout_internal_documents.jsonl")

write_jsonl(train_chunks, proc_chunks_dir / "qasper_train_internal_chunks.jsonl")
write_jsonl(heldout_chunks, proc_chunks_dir / "qasper_heldout_internal_chunks.jsonl")

print(json.dumps({
    "seed": SEED,
    "heldout_frac": HELDOUT_FRAC,
    "num_train_docs": len(train_ids),
    "num_heldout_docs": len(heldout_ids),
    "num_train_queries": len(train_queries),
    "num_heldout_queries": len(heldout_queries),
    "num_train_processed_docs": len(train_docs),
    "num_heldout_processed_docs": len(heldout_docs),
    "num_train_chunks": len(train_chunks),
    "num_heldout_chunks": len(heldout_chunks),
    "unmatched_train_queries": unmatched_train_queries,
    "unmatched_heldout_queries": unmatched_heldout_queries,
    "unmatched_train_docs": unmatched_train_docs,
    "unmatched_heldout_docs": unmatched_heldout_docs,
    "unmatched_train_chunks": unmatched_train_chunks,
    "unmatched_heldout_chunks": unmatched_heldout_chunks,
}, indent=2))
