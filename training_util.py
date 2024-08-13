import os
import json
import random
import pandas as pd


class DataSorter:
    def __init__(self) -> None:
        self.data = dict()

    def from_json(self, folder):
        files = os.listdir(folder)
        for file in files:
            with open(folder + file, "r") as f:
                name = os.path.splitext(file)[0]
                df = pd.read_json()
                self.data[name] = df

    def chunk():
        # Splits data into k even chunks.
        raise NotImplementedError()
