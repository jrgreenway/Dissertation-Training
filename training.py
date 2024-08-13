from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch
import json
import pandas as pd

from training_util import DataSorter

model_name = ""
tokeniser = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokeniser, max_length):
        self.data = data
        self.tokeniser = tokeniser
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, index
    ):  # TODO redo this class to reflect use of DataFrames instead of dictionaries
        item = self.data[index]
        text = (
            f"Primary X: {item['primary']['X']}, Primary Y: {item['primary']['Y']}, "
            f"Primary Heading: {item['primary']['Heading']}, Primary Speed: {item['primary']['Speed']}, "
            f"Secondary X: {item['secondary']['X']}, Secondary Y: {item['secondary']['Y']}, "
            f"Secondary Heading: {item['secondary']['Heading']}, Secondary Speed: {item['secondary']['Speed']}"
        )
        tokens = self.tokeniser(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"].flatten(),
            "attention_mask": tokens["attention_mask"].flatten(),
            "label": item["label"],
        }


data = DataSorter()


max_length = 128
dataset = Dataset(data, tokeniser, max_length)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
