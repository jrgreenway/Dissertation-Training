import os
import json
import pandas as pd
from GLOBAL_VAR import TESTING
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from transformers import AutoModelForSequenceClassification


class DataSorter:
    def __init__(self, fraction=1.0) -> None:
        """Fraction is the fraction of all data to collate for getting smaller datasets"""
        self.data = dict()
        self.length = None
        self.fraction = fraction

    def _info(self, df: pd.DataFrame):
        print(df.describe())
        print(df.head())

    def from_json(self, folder):
        files = os.listdir(folder)
        for file in files:
            with open(folder + file, "r") as f:
                name = os.path.splitext(file)[0]
                df = pd.read_json(f)
                if self.fraction < 1.0:
                    df = df.sample(frac=self.fraction)
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
        self.metrics = {}
        self.accuracy = np.array([])
        self.precision = np.array([])
        self.recall = np.array([])
        self.f1 = np.array([])
        self.confusion_matrix = np.array([])
        self.result_bool = False

    def add_metrics(self, fold, epoch, a, p, r, f1, conf):
        if fold not in self.metrics:
            self.metrics[fold] = {}
        self.metrics[fold][epoch] = {
            "accuracy": a,
            "precision": p,
            "recall": r,
            "f1": f1,
            "conf_matrix": conf.tolist(),  # Convert numpy array to list for JSON serialization
        }

    def end(self, folder):
        self.result_bool = True
        self.save(folder)

    def save(self, folder):
        if self.result_bool:
            with open(f"{folder}result_metrics.json", "w") as f:
                json.dump(self.result(), f)
        with open(f"{folder}metrics.json", "w") as f:
            json.dump(self.metrics, f)

    def result(self):
        aggregated_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        for fold in self.metrics:
            for epoch in self.metrics[fold]:
                aggregated_metrics["accuracy"].append(
                    self.metrics[fold][epoch]["accuracy"]
                )
                aggregated_metrics["precision"].append(
                    self.metrics[fold][epoch]["precision"]
                )
                aggregated_metrics["recall"].append(self.metrics[fold][epoch]["recall"])
                aggregated_metrics["f1"].append(self.metrics[fold][epoch]["f1"])

        return aggregated_metrics


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
            f"The primary ship has a position ({int(row['X_1'])}, {int(row['Y_1'])}), "
            f"a heading of {int(row['Heading_1'])} degrees, and a speed of {int(row['Speed_1'])} knots. "
            f"The secondary ship has a position ({int(row['X_2'])}, {int(row['Y_2'])}), "
            f"a heading of {int(row['Heading_2'])} degrees, and a speed of {int(row['Speed_2'])} knots."
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


class Trainer:
    def __init__(
        self, model_name, metrics, tokeniser, device, batch_size, max_length, labels
    ):
        self.model_name = model_name
        self.metrics = metrics
        self.tokeniser = tokeniser
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.labels = labels

    def _make_model(self):
        m = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.labels
        )
        o = optim.Adam(m.parameters(), lr=1e-5)
        return m, o

    def k_fold_train(
        self,
        k_data,
        metric_save_folder: str,
        model_save_folder: str,
        tokeniser_save_folder: str,
        num_epochs: int,
    ):
        train_losses = []
        val_losses = []
        for fold, (t_data, v_data) in enumerate(k_data):
            fold_t_loss = []
            fold_v_loss = []
            logging.info(f"Fold {fold+1} / {len(k_data)}")
            model, optimiser = self._make_model()
            model.to(self.device)

            for epoch in range(num_epochs):
                logging.info(f"Epoch {epoch+1} / {num_epochs}")
                model.train()
                train_loss = 0.0
                dataset = Dataset(t_data, self.tokeniser, self.max_length)
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=self.batch_size, shuffle=True
                )

                for batch in dataloader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    optimiser.step()
                    optimiser.zero_grad()

                    train_loss += loss.item()

                train_loss /= len(dataloader)
                fold_t_loss.append(train_loss)
                model.eval()
                val_loss = 0.0
                all_labels = []
                all_predictions = []
                with torch.no_grad():
                    dataset = Dataset(v_data, self.tokeniser, self.max_length)
                    dataloader = torch.utils.data.DataLoader(
                        dataset, batch_size=self.batch_size, shuffle=False
                    )
                    for batch in dataloader:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        labels = batch["labels"]
                        outputs = model(**batch)
                        loss = outputs.loss
                        val_loss += loss.item()
                        logits = outputs.logits
                        _, preds = torch.max(logits, dim=1)
                        all_labels.extend(labels.cpu().numpy())
                        all_predictions.extend(preds.cpu().numpy())

                val_loss /= len(dataloader)
                fold_v_loss.append(val_loss)
                accuracy = (np.array(all_labels) == np.array(all_predictions)).mean()
                precision = precision_score(
                    all_labels, all_predictions, average="weighted", zero_division=0
                )
                recall = recall_score(all_labels, all_predictions, average="weighted")
                f1 = f1_score(all_labels, all_predictions, average="weighted")
                conf_matrix = confusion_matrix(all_labels, all_predictions)

                self.metrics.add_metrics(
                    fold, epoch, accuracy, precision, recall, f1, conf_matrix
                )

                logging.info(
                    f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
                )
                logging.info(
                    f"Validation Accuracy for fold {fold+1}, epoch {epoch+1}: {accuracy:.4f}"
                )
                logging.info(
                    f"Validation Precision for fold {fold+1}, epoch {epoch+1}: {precision:.4f}"
                )
                logging.info(
                    f"Validation Recall for fold {fold+1}, epoch {epoch+1}: {recall:.4f}"
                )
                logging.info(
                    f"Validation F1 Score for fold {fold+1}, epoch {epoch+1}: {f1:.4f}"
                )

                self.metrics.save(metric_save_folder)

            model.save_pretrained(f"{model_save_folder}fold_{fold}")
            self.tokeniser.save_pretrained(f"{tokeniser_save_folder}fold_{fold}")

            train_losses.append(fold_t_loss)
            val_losses.append(fold_v_loss)
