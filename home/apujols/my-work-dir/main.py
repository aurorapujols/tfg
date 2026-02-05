import argparse
import yaml
import os

# Experiment entry points
from experiments.run_ssl import run_ssl_experiment


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Meteor Representation Learning Pipeline")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["ssl", "supcon", "classifier", "clustering"],
        help="Which experiment to run"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    # Load YAML config
    cfg = load_config(args.config)

    # Dispatch to the correct experiment
    if args.task == "ssl":
        run_ssl_experiment(cfg)

    elif args.task == "supcon":
        run_supcon_experiment(cfg)

    elif args.task == "classifier":
        run_classifier_experiment(cfg)

    elif args.task == "clustering":
        run_clustering_experiment(cfg)


if __name__ == "__main__":
    main()
