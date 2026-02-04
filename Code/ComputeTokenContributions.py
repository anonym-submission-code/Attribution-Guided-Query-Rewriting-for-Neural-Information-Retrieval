from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from beir.datasets.data_loader import GenericDataLoader
from beir import util
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

#load dataset
dataset = "scifact"

data_path = util.download_and_unzip(
    f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip",
    "datasets")

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")


#load IR Model
model_path = "models/tct_colbert-v2"  
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)
model.eval()

# SPLADE encoding functions
def encode_query_per_token(query_text, max_len=128):
    """
    Returns:
      - tokens: list of token strings
      - token_sparse_vectors: per-token SPLADE sparse vector
    """
    inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        outputs = model(**inputs)  # logits: [1, seq_len, vocab_size]
    
    logits = outputs.logits[0]  # [seq_len, vocab_size]
    token_sparse_vectors = torch.log1p(F.relu(logits))  # SPLADE transformation
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens, token_sparse_vectors

def encode_query(query_text, max_len=128):
    tokens, token_sparse_vectors = encode_query_per_token(query_text, max_len)
    # max pooling over tokens
    return torch.max(token_sparse_vectors, dim=0).values

def encode_document(doc_text, max_len=512):
    inputs = tokenizer(doc_text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    return torch.max(torch.log1p(F.relu(logits)), dim=0).values

def compute_splade_token_contributions_batch(queries, corpus, top_k_docs=5):
    """
    queries: dict {qid: query_text}
    corpus: dict {did: {"text": ...}}
    top_k_docs: nombre de docs à considérer pour les contributions
    """
    # ---- 1) Encoder tout le corpus ----
    print("Encoding corpus...")
    doc_vectors = {}
    for did, doc in corpus.items():
        doc_vectors[did] = encode_document(doc["text"])  # [vocab_dim]
    print("Corpus encoded.")

    # ---- 2) Encoder toutes les requêtes ----
    query_vectors = {}
    query_tokens = {}
    query_token_sparse = {}

    print("Encoding queries...")
    for qid, q_text in queries.items():
        query_vectors[qid] = encode_query(q_text)             # [vocab_dim]
        tokens, tok_sparse = encode_query_per_token(q_text)
        query_tokens[qid] = tokens
        query_token_sparse[qid] = tok_sparse                  # [seq_len, vocab_dim]
    print("Queries encoded.")

    # ---- 3) Scores et contributions ----
    token_attributions = {}
    top_docs_per_query = {}

    for qid in queries.keys():
        print("Q-"+qid)
        q_vec = query_vectors[qid]
        tok_sparse = query_token_sparse[qid]
        tokens = query_tokens[qid]

        # dot-product avec tous les docs
        doc_scores = {did: float(torch.dot(q_vec, d_vec)) for did, d_vec in doc_vectors.items()}

        # top-K documents
        top_docs = sorted(doc_scores.items(), key=lambda x: -x[1])[:top_k_docs]
        top_docs_per_query[qid] = top_docs

        # contributions par token (somme sur top-K docs)
        contrib_sum = np.zeros(len(tokens))
        for did, _ in top_docs:
            d_vec = doc_vectors[did]
            for i, tok_vec in enumerate(tok_sparse):
                contrib_sum[i] += float(torch.dot(tok_vec, d_vec))

        token_attributions[qid] = {
            "tokens": tokens,
            "contributions": contrib_sum
        }

    return token_attributions, top_docs_per_query


# TCT-ColBERT encoding functions
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

import numpy as np
import torch

@torch.no_grad()
def compute_tctColBERT_token_contributions_batch(
    queries,
    corpus,
    top_k_docs=5
):
    """
    queries: dict {qid: query_text}
    corpus: dict {did: {"text": ...}}
    top_k_docs: nombre de docs considérés pour les contributions
    """

    # =========================
    # 1) Encoder le corpus
    # =========================
    print("Encoding corpus (TCT-ColBERT)...")
    doc_embeddings = {}
    for did, doc in corpus.items():
        # [D_i, dim]
        doc_embeddings[did] = encode_document(doc["text"])
    print("Corpus encoded.")

    # =========================
    # 2) Encoder les requêtes
    # =========================
    print("Encoding queries (TCT-ColBERT)...")
    query_embeddings = {}
    query_tokens = {}

    for qid, q_text in queries.items():
        inputs = tokenizer(
            q_text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)

        outputs = model(**inputs)
        emb = outputs.last_hidden_state.squeeze(0)
        mask = inputs["attention_mask"].squeeze(0).bool()

        query_embeddings[qid] = emb[mask]     # [Q, dim]
        query_tokens[qid] = tokenizer.convert_ids_to_tokens(
            inputs["input_ids"].squeeze(0)[mask]
        )

    print("Queries encoded.")

    # =========================
    # 3) Scoring + contributions
    # =========================
    token_attributions = {}
    top_docs_per_query = {}

    for qid in queries.keys():
        print(f"Q-{qid}")
        q_emb = query_embeddings[qid]     # [Q, dim]
        tokens = query_tokens[qid]
        n_tokens = q_emb.size(0)

        # ----- Score documents (ColBERT MaxSim) -----
        doc_scores = {}
        per_doc_token_scores = {}

        for did, d_emb in doc_embeddings.items():
            # [Q, D]
            sim = torch.matmul(q_emb, d_emb.T)
            max_sim, _ = sim.max(dim=1)      # [Q]

            doc_scores[did] = max_sim.sum().item()
            per_doc_token_scores[did] = max_sim.cpu().numpy()

        # ----- Top-K documents -----
        top_docs = sorted(
            doc_scores.items(),
            key=lambda x: -x[1]
        )[:top_k_docs]

        top_docs_per_query[qid] = top_docs

        # ----- Token contributions -----
        contrib_sum = np.zeros(n_tokens)
        for did, _ in top_docs:
            contrib_sum += per_doc_token_scores[did]

        token_attributions[qid] = {
            "query": queries[qid],
            "tokens": tokens,
            "contributions": contrib_sum
        }

    return token_attributions, top_docs_per_query


#main

token_attributions, top_docs = compute_[Model_name]_token_contributions_batch(queries, corpus, top_k_docs=5)













































































































