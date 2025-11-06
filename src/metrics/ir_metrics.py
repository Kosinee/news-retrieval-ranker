import numpy as np

def recall_at_k(qrels, results, k=10):
    hits, total = 0, 0
    for q, rels in qrels.items():
        relevant_docs = set(rels.keys())
        retrieved = set(results.get(q, [])[:k])
        hits += len(relevant_docs & retrieved)
        total += len(relevant_docs)
    return hits / max(total, 1)


def ndcg_at_k(qrels, results, k=10):
    def dcg(labels):
        return np.sum((2 ** labels - 1) / np.log2(np.arange(2, 2 + len(labels))))
    vals = []
    for q, rels in qrels.items():
        relevant_docs = set(rels.keys())
        ranked = results.get(q, [])[:k]
        labels = np.array([1.0 if d in relevant_docs else 0.0 for d in ranked])
        if not np.any(labels):
            continue
        ideal = np.sort(labels)[::-1]
        vals.append(dcg(labels) / (dcg(ideal) + 1e-12))
    return float(np.mean(vals) if vals else 0.0)


def compute_metrics(qrels, results, k_vals=(10, 100)):
    metrics = {}
    for k in k_vals:
        metrics[f"recall@{k}"] = recall_at_k(qrels, results, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(qrels, results, k)
    return metrics
