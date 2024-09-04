from collections import defaultdict
from benchmark_utils import *
from training_util import Dataset
from torch.utils.data import DataLoader
from data_gen import gen_crossing_group
from data_types import Situation, Event, EventGroup
from GLOBAL_VAR import CONSTRAINTS
import json


eventgroup = gen_crossing_group(10, Situation.CROSSING, CONSTRAINTS)
model_dict = defaultdict(
    lambda: "bert", {"bert": "bert", "distilbert": "distilbert", "xlnet": "xlnet"}
)


def benchmarker(
    model_name: str,
    data,
):
    model, tokeniser = start_model(model_name)
    data = Dataset(data, tokeniser, 128)
    dataloader = DataLoader(data, batch_size=1, shuffle=False)

    total_time = 0
    num_sentences = 0

    for batch in dataloader:
        inputs = {
            "input_ids": batch["input_ids"].to(model.device),
            "attention_mask": batch["attention_mask"].to(model.device),
        }
        outputs, elapsed_time = benchmark_model(model, inputs)
        total_time += elapsed_time
        num_sentences += 1

    avg_time_per_sentence = total_time / num_sentences
    print(f"Average time per sentence: {avg_time_per_sentence:.4f} seconds")
    return avg_time_per_sentence


models = ["bert", "distilbert", "xlnet"]


for model in models:
    model_dict[model] = benchmarker(model, eventgroup.group)

with open("benchmark.json", "w") as json_file:
    json.dump(model_dict, json_file)
