"""Utilities for loading workflow configuration."""

from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load a YAML configuration file."""

    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


if __name__ == "__main__":
    config = load_config("config/west_siberian_arctic.yaml")

    print("Project:", config["project"]["name"])
    print("Region:", config["project"]["region"])
    print("AOI:", config["paths"]["aoi"])
    print(
        "ArcticDEM resolution:",
        config["processing"]["arcticdem"]["resolution_m"],
    )
    print("Workers:", config["compute"]["n_workers"])