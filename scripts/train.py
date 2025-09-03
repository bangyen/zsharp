#!/usr/bin/env python3
"""Simple training script for running individual experiments."""

import argparse
import logging
import sys

import yaml

from src.train import train


def main():
    parser = argparse.ArgumentParser(
        description="Train a model with ZSharp or SGD"
    )
    parser.add_argument(
        "--config", required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    args = parser.parse_args()

    # Setup logging without prefix
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s")
    logger = logging.getLogger(__name__)

    # Load configuration
    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Error: Configuration file '{args.config}' not found")
        return 1
    except yaml.YAMLError as e:
        logger.error(f"Error: Invalid YAML in configuration file: {e}")
        return 1

    # Run training
    try:
        results = train(config)
        logger.info("Training completed!")
        logger.info(
            "Final test accuracy: {:.2f}%".format(
                results["final_test_accuracy"]
            )
        )
        logger.info(
            "Final test loss: {:.4f}".format(results["final_test_loss"])
        )
        logger.info(
            "Training time: {:.2f} seconds".format(
                results["total_training_time"]
            )
        )
        return 0
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
