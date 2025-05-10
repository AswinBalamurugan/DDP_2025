import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from src.data.graph_manager import load_graph
from src.evaluation.evaluator import run_evaluation
from config.hyperparams import DATASET_NAMES


def main():
    """Main function to run evaluations with improved logging."""
    parser = argparse.ArgumentParser(description="Run link prediction evaluations.")
    parser.add_argument(
        "--output_dir", default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASET_NAMES, help="Datasets to evaluate"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Initialize seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    for dataset_name in tqdm(args.datasets, desc="Evaluating datasets"):
        G = load_graph(dataset_name, processed_dir="data")
        run_evaluation(
            dataset_name=dataset_name,
            G=G,
            output_dir=args.output_dir,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
