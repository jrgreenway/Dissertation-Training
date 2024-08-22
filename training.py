from transformers import AutoTokenizer
import torch.nn as nn
import torch
import torch.optim as optim
from datetime import datetime

from GLOBAL_VAR import TESTING
import logging
import numpy as np
from training_util import DataSorter, Metrics, Dataset, Trainer
import os
import argparse


parser = parser = argparse.ArgumentParser(description="Train a model.")
parser.add_argument(
    "--model", type=str, help="The model to train (bert, xlnet or distilbert)"
)
parser.add_argument(
    "--fraction",
    type=float,
    default=1.0,
    help="The fraction of the dataset to train on (must be <1.0 and >0)",
)
args = parser.parse_args()

model_id = args.model
fraction = args.fraction

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

run_folder = f"/app/results/{model_id}/{timestamp}_{fraction}/"
training_data_folder = "testing/events/" if TESTING else "/app/results/events/"
metric_save_folder = "testing/metrics/" if TESTING else f"{run_folder}metrics/"
model_save_folder = "testing/models/" if TESTING else f"{run_folder}models/"
tokeniser_save_folder = "testing/tokeniser/" if TESTING else f"{run_folder}tokenisers/"

for folder in [metric_save_folder, model_save_folder, tokeniser_save_folder]:
    os.makedirs(folder, exist_ok=True)

# Data Loading
data = DataSorter(fraction=fraction)
data.from_json(training_data_folder)
k_data = data.split(5)

# Model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name = f"{model_id}-base-cased"
tokeniser = AutoTokenizer.from_pretrained(model_name)

logging.info(f"Using device: {device}")

max_length = 128

metrics = Metrics()
trainer = Trainer(model_name, metrics, tokeniser, device, 32, max_length, 3)

trainer.k_fold_train(
    k_data, metric_save_folder, model_save_folder.tokeniser_save_folder, learning=True
)

try:
    metrics.end(metric_save_folder)
    logging.info("End of Script")
except Exception as e:
    logging.error(e)
