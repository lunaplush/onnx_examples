import numpy as np
import pandas as pd
import time
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer


if __name__ == "__main__":
    model_name = "ElKulako/cryptobert"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=64,
                                              truncation=True, padding='max_length', top_k=None)

    start_time = time.time()
    print(pipe("There is positive news about bitcoin"))
    print(time.time()-start_time)