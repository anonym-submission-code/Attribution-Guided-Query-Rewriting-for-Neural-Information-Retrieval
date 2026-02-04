from sentence_transformers import SentenceTransformer, losses, models
from transformers import AutoModel, AutoTokenizer
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import logging
import pathlib
import torch
import os

os.environ["WANDB_DISABLED"] = "true"


device = "cuda" if torch.cuda.is_available() else "cpu"


data_path = "datasets/fiqa"

# Provide the data_path where the dataset has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")


## Provide the sentence-transformer or HF model
model_name = "models/tct_colbert-v2-msmarco"
word_embedding_model = models.Transformer(model_name, max_seq_length=350)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])



retriever = TrainRetriever(model=model, batch_size=16)

#### Prepare training samples
train_samples = retriever.load_train(corpus, queries, qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

## Training SBERT with cosine-product
train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)

# training SBERT with dot-product
# train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score)

#### Prepare dev evaluator
ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)


def main():
    #### Provide model save path
    model_save_path = "models/tct_colbert-v2-finetuned-fiqa"
    os.makedirs(model_save_path, exist_ok=True)
    
    #### Configure Train params
    num_epochs = 5
    evaluation_steps = 5000
    warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)
    
    retriever.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=ir_evaluator,
        epochs=num_epochs,
        output_path=model_save_path,
        warmup_steps=warmup_steps,
        evaluation_steps=evaluation_steps,
        use_amp=True,
    )

if __name__ == "__main__":
    main()





















































