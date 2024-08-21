from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn
import torch
import torch.optim as optim
from datetime import datetime

from GLOBAL_VAR import TESTING
import logging
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from training_util import DataSorter, Metrics, Dataset
import os


model_id = "distilbert"
log_file_path = f"/app/logs/{model_id}-training.log"

now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")

if os.path.exists(log_file_path):
    os.remove(log_file_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
    ],
)


training_data_folder = "testing/events/" if TESTING else "/app/results/events/"
metric_save_folder = (
    "testing/metrics/" if TESTING else f"/app/results/{model_id}/metrics/"
)
model_save_folder = "testing/models/" if TESTING else f"/app/results/{model_id}/models/"

for folder in [metric_save_folder, model_save_folder]:
    os.makedirs(folder, exist_ok=True)


def make_model(name, labels=3):
    m = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=labels
    )
    o = optim.Adam(m.parameters(), lr=1e-5)
    return m, o


# Data Loading
data = DataSorter()
data.from_json(training_data_folder)
k_data = data.split(5)

# Model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name = f"{model_id}-base-cased"
tokeniser = AutoTokenizer.from_pretrained(model_name)

logging.info(f"Using device: {device}")

max_length = 128

metrics = Metrics()

for fold, (t_data, v_data) in enumerate(k_data):
    logging.info(f"Fold {fold+1} / {len(k_data)}")
    model, optimiser = make_model(model_name)
    model.to(device)

    model.train()

    dataset = Dataset(t_data, tokeniser, max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        # logging.info(f"Loss: {loss.item()}")

    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        dataset = Dataset(v_data, tokeniser, max_length)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            outputs = model(**batch)
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

    accuracy = (np.array(all_labels) == np.array(all_predictions)).mean()
    precision = precision_score(
        all_labels, all_predictions, average="weighted", zero_division=0
    )
    recall = recall_score(all_labels, all_predictions, average="weighted")
    f1 = f1_score(all_labels, all_predictions, average="weighted")
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    metrics.add_metrics(accuracy, precision, recall, f1, conf_matrix)

    logging.info(f"Validation Accuracy for fold {fold+1}: {accuracy:.4f}")
    logging.info(f"Validation Precision for fold {fold+1}: {precision:.4f}")
    logging.info(f"Validation Recall for fold {fold+1}: {recall:.4f}")
    logging.info(f"Validation F1 Score for fold {fold+1}: {f1:.4f}")

    metrics.save(metric_save_folder + timestamp + "/")
    model.save_pretrained(f"{model_save_folder}fold_{fold}_{timestamp}")
    tokeniser.save_pretrained(f"{model_save_folder}fold_{fold}_{timestamp}")

metrics.end(metric_save_folder + timestamp + "/")
logging.info("End of Script")
