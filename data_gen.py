import math
import os
import random

from tqdm import tqdm
from GLOBAL_VAR import *
from data_types import *

total_size = CROSSING_SAMPLE + OVERTAKING_SAMPLE + HEAD_ON_SAMPLE
total_count = 0


def gen_primary_ship(constraints):
    c = constraints
    ship = Ship()
    ship.Heading = 0
    ship.Speed = random.uniform(c["speed_min"], c["speed_max"])
    ship.X = c["x_max"] / 2
    ship.Y = c["y_max"] / 2
    return ship


def gen_x_y(x, y, bearing, distance):
    bearing_rad = math.radians(bearing)
    new_x = x + distance * math.cos(bearing_rad)
    new_y = y + distance * math.sin(bearing_rad)
    return new_x, new_y


def crossing_helper(primary_ship, c, ship):
    target_x = primary_ship.X
    lower_y = primary_ship.Y + 20
    upper_y = c["y_max"]

    o0 = lower_y - ship.Y
    a0 = target_x - ship.X
    o1 = upper_y - ship.Y
    a1 = target_x - ship.X

    lower = math.degrees(math.atan2(o0, a0))
    upper = math.degrees(math.atan2(o1, a1))
    lower_angle = (lower + 360) % 360
    upper_angle = (upper + 360) % 360
    if lower_angle > upper_angle:
        lower_angle, upper_angle = upper_angle, lower_angle
    return lower_angle, upper_angle


def gen_random_ship(constraints, primary_ship, situation: Situation) -> Ship:
    c = constraints
    ship = Ship()
    distance = random.uniform(50, c["x_max"] / 2)
    if situation == Situation.OVERTAKE:
        bearing = random.uniform(*OVERTAKE_BEARING)
        ship.X, ship.Y = gen_x_y(primary_ship.X, primary_ship.Y, bearing, distance)
        ship.Heading = 0
        ship.Speed = primary_ship.Speed + random.uniform(0.5, 10)
    elif situation == Situation.HEAD_ON:
        bearing = random.uniform(-HEAD_ON_BEARING[2], HEAD_ON_BEARING[2])
        if bearing < 0:
            bearing = 360 + bearing
        ship.X, ship.Y = gen_x_y(primary_ship.X, primary_ship.Y, bearing, distance)
        ship.Heading = 180
        ship.Speed = random.uniform(c["speed_min"], c["speed_max"])
    elif situation == Situation.CROSSING_1:
        bearing = random.uniform(*CROSSING_BEARING_1)
        ship.X, ship.Y = gen_x_y(primary_ship.X, primary_ship.Y, bearing, distance)
        lower, upper = crossing_helper(primary_ship, c, ship)
        ship.Heading = random.uniform(lower, upper)

        if bearing < 90:
            speed1 = c["speed_min"]
            speed2 = c["speed_max"]
        else:
            speed1 = primary_ship.Speed
            speed2 = c["speed_max"] + 5
        ship.Speed = random.uniform(speed1, speed2)
    elif situation == Situation.CROSSING_2:
        bearing = random.uniform(*CROSSING_BEARING_2)
        ship.X, ship.Y = gen_x_y(primary_ship.X, primary_ship.Y, bearing, distance)
        lower, upper = crossing_helper(primary_ship, c, ship)
        ship.Heading = random.uniform(lower, upper)

        if bearing > 270:
            speed1 = primary_ship.Speed
            speed2 = c["speed_max"] + 5
        else:
            speed1 = c["speed_min"]
            speed2 = c["speed_max"]
        ship.Speed = random.uniform(speed1, speed2)
    return ship


def gen_eventgroup(size, situation: Situation, constraints):
    group = EventGroup()
    for i in tqdm(range(size)):
        event = Event()
        primary_ship = gen_primary_ship(constraints)
        secondary_ship = gen_random_ship(constraints, primary_ship, situation)
        event.label = situation
        event.ship1 = primary_ship
        event.ship2 = secondary_ship
        group.add_event(event)
    return group


if TESTING:
    save_path = "testing/events/"
else:
    save_path = "results/events/"

for i in [save_path]:
    os.makedirs(i)

for sit, sample in zip(
    list(Situation),
    [OVERTAKING_SAMPLE, HEAD_ON_SAMPLE, CROSSING_SAMPLE / 2, CROSSING_SAMPLE / 2],
):
    eventgroup = gen_eventgroup(sample, sit, CONSTRAINTS)
    eventgroup.to_json(save_path, sit.name)
