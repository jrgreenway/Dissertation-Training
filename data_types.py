from enum import Enum
import json


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

    def to_dict(self):
        return dict(
            primary=self.ship1.to_dict(),
            secondary=self.ship2.to_dict(),
            label=self.label.value,
        )


class EventGroup:
    def __init__(self):
        self.id_count = 0
        self.group = dict()

    def add_event(self, event: Event):
        self.group[self.id_count] = event.to_dict()
        self.id_count += 1

    def to_json(self, file_path, name):
        with open(file_path + name + ".json", "w") as file:
            json.dump(self.group, file)


class Situation(Enum):
    OVERTAKE = 0
    HEAD_ON = 1
    CROSSING_1 = 2
    CROSSING_2 = 3
