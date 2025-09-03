#!/usr/bin/env python3
"""Simple training script for running individual experiments."""

import argparse
import logging
import signal
import sys

import yaml

from src.train import train


def signal_handler(_sig, _frame):  # pylint: disable=unused-argument
    """Handle Ctrl+C gracefully by logging and exiting cleanly."""
    logger = logging.getLogger(__name__)
    logger.warning("Interrupted by user. Cleaning up...")
    sys.exit(0)


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

    # Set up signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)

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
        if results is None:
            logger.warning("Training was interrupted by user")
            return 0
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
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
