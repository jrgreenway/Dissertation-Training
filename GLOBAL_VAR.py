CROSSING_SAMPLE = 100000
OVERTAKING_SAMPLE = 100000
HEAD_ON_SAMPLE = 100000

OVERTAKE_BEARING = (160, 200)
HEAD_ON_BEARING = (345, 15, 30)
CROSSING_BEARING_1 = (15, 160)
CROSSING_BEARING_2 = (200, 345)

TESTING = False

CONSTRAINTS = dict(
    x_min=0,
    x_max=1000,
    y_min=0,
    y_max=1000,
    speed_max=20,
    speed_min=0.5,
    heading_max=360,
    heading_min=0,
)
