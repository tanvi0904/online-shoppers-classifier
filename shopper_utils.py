"""Shared bits for training and the Streamlit app: column lists, target parsing, metrics."""

TARGET = "Revenue"

# heavy right-skewed counts/durations -> log1p before scaling
SKEWED_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "PageValues",
]
RATE_COLS = ["BounceRates", "ExitRates", "SpecialDay"]
# integer codes in the raw data, but they are labels, not quantities
CATEGORICAL_COLS = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]

FEATURES = SKEWED_COLS + RATE_COLS + CATEGORICAL_COLS
