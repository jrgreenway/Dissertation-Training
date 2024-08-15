import os
import json
import random
import pandas as pd
from GLOBAL_VAR import TESTING
import numpy as np
from torch.utils.data import Dataset
import torch


class DataSorter:
    def __init__(self) -> None:
        self.data = dict()
        self.length = None

    def _info(self, df: pd.DataFrame):
        print(df.describe())
        print(df.head())

    def from_json(self, folder):
        files = os.listdir(folder)
        for file in files:
            with open(folder + file, "r") as f:
                name = os.path.splitext(file)[0]
                df = pd.read_json(f)
                self.data[name] = df
                if TESTING:
                    self._info(df)
        self.length = len(self.data[list(self.data.keys())[0]])

    def split(self, k):
        if k < 2:
            raise ValueError("k must be at least 2")
        indices = np.arange(self.length)
        np.random.shuffle(indices)
        folds = np.array_split(indices, k)
        kfolds = []
        for i in range(k):
            validation_indices = folds[i]
            train_indices = np.concatenate([folds[j] for j in range(k) if j != i])

            train_data = {
                key: df.iloc[train_indices].reset_index(drop=True)
                for key, df in self.data.items()
            }
            validation_data = {
                key: df.iloc[validation_indices].reset_index(drop=True)
                for key, df in self.data.items()
            }

            concatenated_train_data = pd.concat(train_data.values()).reset_index(
                drop=True
            )
            concatenated_validation_data = pd.concat(
                validation_data.values()
            ).reset_index(drop=True)
            concatenated_train_data = concatenated_train_data.sample(
                frac=1
            ).reset_index(drop=True)
            concatenated_validation_data = concatenated_validation_data.sample(
                frac=1
            ).reset_index(drop=True)
            kfolds.append((concatenated_train_data, concatenated_validation_data))

        return kfolds


class Metrics:
    def __init__(self):
        self.accuracy = np.array([])
        self.precision = np.array([])
        self.recall = np.array([])
        self.f1 = np.array([])
        self.confusion_matrix = np.array([])
        self.result_bool = False

    def to_dict(self):
        return {
            "accuracy": self.accuracy.tolist(),
            "precision": self.precision.tolist(),
            "recall": self.recall.tolist(),
            "f1": self.f1.tolist(),
            "confusion_matrix": self.confusion_matrix.tolist(),
        }

    def add_metrics(self, a, p, r, f1, conf):
        self.accuracy = np.append(self.accuracy, a)
        self.precision = np.append(self.precision, p)
        self.recall = np.append(self.recall, r)
        self.f1 = np.append(self.f1, f1)
        self.confusion_matrix = np.append(self.confusion_matrix, conf)

    def end(self, folder):
        self.result_bool = True
        self.save(folder)

    def save(self, folder):
        if self.result_bool:
            with open(f"{folder}result_metrics.json", "w") as f:
                json.dump(self.result(), f)
        with open(f"{folder}metrics.json", "w") as f:
            json.dump(self.to_dict(), f)

    def result(self):
        avg_accuracy = np.mean(self.accuracy)
        std_accuracy = np.std(self.accuracy)
        avg_precision = np.mean(self.precision)
        std_precision = np.std(self.precision)
        avg_recall = np.mean(self.recall)
        std_recall = np.std(self.recall)
        avg_f1 = np.mean(self.f1)
        std_f1 = np.std(self.f1)
        avg_conf_matrix = np.mean(self.confusion_matrix, axis=0)

        return {
            "avg_accuracy": avg_accuracy,
            "std_accuracy": std_accuracy,
            "avg_precision": avg_precision,
            "std_precision": std_precision,
            "avg_recall": avg_recall,
            "std_recall": std_recall,
            "avg_f1": avg_f1,
            "std_f1": std_f1,
            "avg_conf_matrix": avg_conf_matrix,
        }


class Dataset(Dataset):
    def __init__(self, data, tokeniser, max_length):
        self.data = data
        self.tokeniser = tokeniser
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = (
            f"The primary ship has a position ({row['X_1']}, {row['Y_1']}), "
            f"a heading of {row['Heading_1']} degrees, and a speed of {row['Speed_1']} knots. "
            f"The secondary ship has a position ({row['X_2']}, {row['Y_2']}), "
            f"a heading of {row['Heading_2']} degrees, and a speed of {row['Speed_2']} knots."
        )

        inputs = self.tokeniser.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        label = int(row["Label"])

        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }
