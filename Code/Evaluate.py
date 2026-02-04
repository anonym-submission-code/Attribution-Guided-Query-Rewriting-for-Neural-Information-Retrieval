from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir import util
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

#load dataset
....

#load your model
....

def evaluate(metrics):
    rows = []

    if isinstance(metrics, tuple):
        ndcg, map_score, recall, precision, scores = metrics
        metric_dicts = {"NDCG": ndcg, "MAP": map_score, "Recall": recall, "Precision": precision}
    else:
        metric_dicts = {"NDCG": metrics["ndcg"], "MAP": metrics["map"],
                        "Recall": metrics["recall"], "Precision": metrics["precision"]}

    for metric_name, d in metric_dicts.items():
        for k_name, value in d.items():

            if "@" in str(k_name):
                k_val = str(k_name).split("@")[1]
            else:
                k_val = str(k_name)
            rows.append({
                "k": k_val,
                "Metric": metric_name,
                "Value": value
            })

    df = pd.DataFrame(rows)
    df = df.pivot(index="k", columns="Metric", values="Value")
    df = df.sort_index(key=lambda x: pd.to_numeric(x, errors='coerce'))  
    return df


# SPLADE retriever ------------------------------------------------------------------

class SPLADERetriever:
    def __init__(self, model):
        self.model = model
        self.tokenizer = tokenizer

    def retrieve(self, corpus, queries, top_k=100):
        results = {}
        # Precompute document vectors
        doc_vecs = {}
        for did, doc in corpus.items():
            doc_vecs[did] = encode_document(doc["text"])
        print("All document encoded...!")
        
        for qid, qtext in queries.items():
            q_vec = encode_query(qtext)
            scores = {}
            for did, d_vec in doc_vecs.items():
                scores[did] = float(torch.dot(q_vec, d_vec))
            # top-k
            top_docs = dict(sorted(scores.items(), key=lambda x: -x[1])[:top_k])
            results[qid] = top_docs
        return results

retriever = SPLADERetriever(model)

# TCT_ColBERT retriever -----------------------------------------------------------------
@torch.no_grad()
def encode_query(query_text):
    inputs = tokenizer(
        query_text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    outputs = model(**inputs)
    # [n_tokens, dim]
    embeddings = outputs.last_hidden_state.squeeze(0)
    mask = inputs["attention_mask"].squeeze(0).bool()

    return embeddings[mask]

@torch.no_grad()
def encode_document(doc_text):
    inputs = tokenizer(
        doc_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0)
    mask = inputs["attention_mask"].squeeze(0).bool()

    return embeddings[mask]

def colbert_score(q_emb, d_emb):
    """
    q_emb: [Q, D]
    d_emb: [D, D]
    """
    sim = torch.matmul(q_emb, d_emb.T)   # [Q, D]
    return sim.max(dim=1).values.sum().item()

class TCTColBERTRetriever:
    def __init__(self, model):
        self.model = model
        self.tokenizer = tokenizer

    def retrieve(self, corpus, queries, top_k=100):
        results = {}

        # 🔹 Precompute document embeddings
        doc_embeddings = {}
        for did, doc in corpus.items():
            doc_embeddings[did] = encode_document(doc["text"]) 

        # 🔹 Score queries
        for qid, qtext in queries.items():
            print(qid)
            q_emb = encode_query(qtext) 
            scores = {}

            for did, d_emb in doc_embeddings.items():
                scores[did] = colbert_score(q_emb, d_emb)

            # Top-k
            results[qid] = dict(
                sorted(scores.items(), key=lambda x: -x[1])[:top_k]
            )

        return results

retriever = TCTColBERTRetriever(model)

#----------------------------------------------------------------------------------
# Run retrieval

top_k = 100

results = retriever.retrieve(corpus, queries, top_k=top_k)

print("Retrieval done!")

metrics = EvaluateRetrieval.evaluate(qrels, results, [1,3,5,10,100])

evaluate(metrics)


























