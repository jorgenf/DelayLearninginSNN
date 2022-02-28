import numpy as np


REPETITIVE_LENGTH = 5000
CONVERGENT_LENGTH = 5000
DIVERGENT_LENGTH = 2000
SATURATION_LENGTH = 5000
STD_THRESHOLD = 0.1
CORRELATION_THRESHOLD = 0.95
SLOPE_THRESHOLD = 0.001
MAX_DELAY = 40
MIN_DELAY = 0.1
COLORS = ["r", "b", "g", "y", "royalblue", "peru", "palegreen", "indigo"]
COLORS += [(np.random.random(), np.random.random(), np.random.random()) for x in range(50)]
DCAT_COLORS = ["green", "red", "blue","yellow"]
DELAY_CATEGORIES = ["converging", "repeating", "uncategorized", "increasing", "decreasing", "min", "max"]
DELAY_CATEGORIES_SHORTLIST = ["converging", "diverging", "repeating", "uncategorized"]
CATEGORY_CONVERSION = {
    "converging": "converging",
    "repeating": "repeating",
    "uncategorized": "uncategorized",
    "max": "diverging",
    "min": "diverging",
    "decreasing": "diverging",
    "increasing": "diverging"
}
