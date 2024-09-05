import torch
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def start_model(model_name):
    # Initialises a presaved model and tokeniser and returns them.
    model_name = f"models/{model_name}"
    tokeniser = AutoTokenizer.from_pretrained(model_name + "_token")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokeniser


def benchmark_model(model, inputs):
    # Returns the time it takes for a model to return outputs.
    model.eval()
    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
        end = time.time()
    return outputs, end - start
