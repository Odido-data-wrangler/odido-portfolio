#!/usr/bin/env python3
"""
Nova Dataset Creator (showcase)

Creates Langfuse datasets from tests/data/test_cases.json for Nova evaluation.
This is a presentation script; it references internal secrets/infra by design
and is not intended to run in this public repository.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add evaluation directory to path
eval_dir = Path(__file__).parent
sys.path.insert(0, str(eval_dir))

import evallib as ev  # type: ignore

# Try to import tomllib (Python 3.11+) or tomli as fallback
try:
	import tomllib  # type: ignore
except ModuleNotFoundError:
	try:
		import tomli as tomllib  # type: ignore
	except ModuleNotFoundError:
		tomllib = None  # type: ignore

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_type: str) -> dict:
	"""Load configuration from TOML file."""
	config_path = eval_dir / "configuration" / f"{config_type}.toml"
	if not config_path.exists():
		raise FileNotFoundError(f"Config not found: {config_path}")
	if tomllib is None:
		raise ImportError("TOML parser not available. Install tomli for py<3.11")
	with open(config_path, "rb") as f:
		config = tomllib.load(f)  # type: ignore[arg-type]
	return config


def create_benchmark_dataset() -> str:
	logger.info("Creating Nova benchmark dataset...")
	benchmark_config = load_config("benchmark")
	langfuse = ev.get_nova_langfuse_client()
	dataset_name = benchmark_config["experiment"]["dataset_name"]
	dataset_description = benchmark_config["experiment"]["dataset_description"]
	ev.create_nova_dataset_from_test_cases(langfuse, dataset_name=dataset_name, description=dataset_description)
	logger.info("✅ Benchmark dataset created")
	return dataset_name


def create_experiment_dataset() -> str:
	logger.info("Creating Nova experiment dataset...")
	experiment_config = load_config("experiment")
	langfuse = ev.get_nova_langfuse_client()
	dataset_name = experiment_config["experiment"]["dataset_name"]
	dataset_description = experiment_config["experiment"]["dataset_description"]
	ev.create_nova_dataset_from_test_cases(langfuse, dataset_name=dataset_name, description=dataset_description)
	logger.info("✅ Experiment dataset created")
	return dataset_name


def create_custom_dataset(dataset_name: str, description: str) -> str:
	logger.info(f"Creating custom Nova dataset: {dataset_name}")
	langfuse = ev.get_nova_langfuse_client()
	ev.create_nova_dataset_from_test_cases(langfuse, dataset_name=dataset_name, description=description)
	logger.info("✅ Custom dataset created")
	return dataset_name


def list_datasets() -> None:
	logger.info("Listing available datasets… (use Langfuse UI in this showcase)")
	# In this showcase, we direct reviewers to the Langfuse UI.
	logger.info("📊 View datasets in Langfuse UI")


def main() -> None:
	parser = argparse.ArgumentParser(description="Nova Dataset Creator - Langfuse datasets from test_cases.json")
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--benchmark', action='store_true', help='Create benchmark dataset from benchmark.toml')
	group.add_argument('--experiment', action='store_true', help='Create experiment dataset from experiment.toml')
	group.add_argument('--custom', nargs=2, metavar=('NAME', 'DESCRIPTION'), help='Create custom dataset')
	group.add_argument('--list', action='store_true', help='List available datasets (UI)')
	args = parser.parse_args()

	if args.benchmark:
		create_benchmark_dataset()
	elif args.experiment:
		create_experiment_dataset()
	elif args.custom:
		create_custom_dataset(args.custom[0], args.custom[1])
	elif args.list:
		list_datasets()


if __name__ == "__main__":  # pragma: no cover
	main()
