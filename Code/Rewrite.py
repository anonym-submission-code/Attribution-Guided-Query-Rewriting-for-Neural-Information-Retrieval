rom transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


#Load attribution file (see ComputeTokenContributions.py)
with open("tct-colbert.scifact.token_attributions.json", "r", encoding="utf-8") as f:
    data = json.load(f)


def rewrite_queries_with_llm(data):
    llm_model_name="mistralai/Mistral-7B-Instruct-v0.3"
    forbidden_tokens={"[CLS]", "[SEP]"}
    max_new_tokens=120


    # 1) Load LLM
    tokenizer_llm = AutoTokenizer.from_pretrained(llm_model_name)
    model_llm = AutoModelForCausalLM.from_pretrained(llm_model_name, device_map="auto", torch_dtype="auto" )

    generator = pipeline("text-generation", model=model_llm, tokenizer=tokenizer_llm)

    rewritten_queries = {}

    for key, value in data.items():
        original_query = value["query"]
        tokens = value["tokens"]
        contribs = np.array(value["contributions"])

        # 2) Filter out [CLS] / [SEP]
        filtered = [
            (tok, float(c))
            for tok, c in zip(tokens, contribs)
            if tok not in forbidden_tokens
        ]

        if not filtered:
            rewritten_queries[key] = original_query
            continue

        filtered_tokens, filtered_contribs = zip(*filtered)

        # 3) Build prompt
        prompt = f"""
You are given:
1) An original user query.
2) A list of query tokens with their attribution scores, where higher scores indicate a stronger positive contribution to retrieval effectiveness, and lower or negative scores indicate weak or misleading contributions.

Your task is to rewrite the query to improve retrieval effectiveness.

Guidelines:
- Preserve the original user intent.
- Do not remove important concepts.
- Tokens with high attribution scores should be preserved or emphasized.
- Tokens with low or negative attribution scores may be clarified, specified, or disambiguated.
- Avoid adding new concepts that are not implied by the original query.
- Produce a single rewritten query, concise and well-formed.

Original query: "{original_query}"

Token attributions: {dict(zip(filtered_tokens, filtered_contribs))}
"""

        # 4) Generate
        response = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        rewritten_query = response[0]["generated_text"].strip()
        #print(rewritten_query)
        rewritten_query = rewritten_query.rsplit('"', 2)[1]
        print(rewritten_query)
        rewritten_queries[key] = rewritten_query
    return rewritten_queries

rewritten_llm_queries = rewrite_queries_with_llm(data)

#Export to JSON file
with open("tct-colbert.scifact.rewritten_llm_queries.json", "w", encoding="utf-8") as f:
    json.dump(rewritten_llm_queries, f, ensure_ascii=False, indent=2)

print(f"LLM rewritten queries exportées!")












