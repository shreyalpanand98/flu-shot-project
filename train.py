import csv
from collections import Counter, defaultdict
from typing import Dict


def build_feature_value_frequencies(
    csv_path: str, missing_token: str = "__MISSING__"
) -> Dict[str, Dict[str, int]]:
    """
    Return a nested dictionary with observed value frequencies per feature.

    Output format:
      {
        "feature_name": {
          "feature_value_1": frequency,
          "feature_value_2": frequency,
          ...
        },
        ...
      }
    """
    feature_counters = defaultdict(Counter)

    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            for feature, value in row.items():
                normalized_value = missing_token if value in (None, "") else str(value)
                feature_counters[feature][normalized_value] += 1

    # Convert defaultdict/Counter objects into plain dictionaries.
    return {feature: dict(counter) for feature, counter in feature_counters.items()}


if __name__ == "__main__":
    frequencies = build_feature_value_frequencies("data/training_set_features.csv")
    for key in frequencies:
        print(key, frequencies[key], "\n")
