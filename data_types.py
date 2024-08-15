from enum import Enum
import json
import pandas as pd


class Ship:
    def __init__(self):
        self.X: float
        self.Y: float
        self.Heading: float
        self.Speed: float

    def to_dict(self):
        return dict(X=self.X, Y=self.Y, Heading=self.Heading, Speed=self.Speed)


class Event:
    def __init__(self) -> None:
        self.ship1: Ship
        self.ship2: Ship
        self.label: Situation

    def to_df(self):
        return

    def to_dict(self):
        return dict(
            primary=self.ship1.to_dict(),
            secondary=self.ship2.to_dict(),
            label=self.label.value,
        )


class EventGroup:
    def __init__(self):
        self.group = None

    def merge_dicts(self, dict1, dict2):
        merged_dict = {}
        for key in dict1:
            merged_dict[f"{key}_1"] = dict1[key]
        for key in dict2:
            merged_dict[f"{key}_2"] = dict2[key]
        return merged_dict

    def add_event(self, event: Event):
        primary_dict = event.ship1.to_dict()
        secondary_dict = event.ship2.to_dict()
        row = self.merge_dicts(primary_dict, secondary_dict)
        row["Label"] = event.label.value
        if self.group is None:
            self.group = pd.DataFrame([row])
        else:
            self.group = pd.concat([self.group, pd.DataFrame([row])], ignore_index=True)

    def to_json(self, file_path, name):
        json_str = self.group.to_json(orient="records")
        with open(file_path + name + ".json", "w") as file:
            file.write(json_str)


class Situation(Enum):
    OVERTAKE = 0
    HEAD_ON = 1
    CROSSING = 2


class Crossing(Enum):
    CROSSING_1 = 0
    CROSSING_2 = 1
