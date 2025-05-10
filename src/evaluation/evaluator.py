from src.algorithms.imports import *
import json
import time
from typing import Any, Dict
from src.algorithms.enhanced_predictor import EnhancedLinkPredictor
from config.hyperparams import VIZ_CONFIG, EVALUATION_CONFIG


def _log_to_json(output_dir: str, log_data: Dict[str, Any]) -> None:
    """Helper to append logs directly to progress.json"""
    log_file = os.path.join(output_dir, "progress.json")
    try:
        with open(log_file, "r+") as f:
            try:
                existing_logs = json.load(f)
            except json.JSONDecodeError:
                existing_logs = []
            existing_logs.append(log_data)
            f.seek(0)
            json.dump(existing_logs, f, indent=2)
    except FileNotFoundError:
        with open(log_file, "w") as f:
            json.dump([log_data], f, indent=2)


def _prune_graph(
    G: nx.Graph,
    output_dir: str,
    dataset_name: str,
    max_nodes: int = 5000,
    max_edges: int = 20000,
) -> nx.Graph:
    """
    Prune the graph to meet node/edge limits with logging to progress_log.json.

    Parameters:
        G: Input graph
        output_dir: Directory for output files
        dataset_name: Name of the dataset
        max_nodes: Maximum allowed nodes
        max_edges: Maximum allowed edges

    Returns:
        Pruned graph
    """
    original_stats = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}

    if G.number_of_nodes() <= max_nodes and G.number_of_edges() <= max_edges:
        _log_to_json(
            output_dir,
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": dataset_name,
                "event": "prune_skip",
                "original_nodes": original_stats["nodes"],
                "original_edges": original_stats["edges"],
            },
        )
        return G

    # Node pruning
    nodes = list(G.nodes())
    if len(nodes) > max_nodes:
        _log_to_json(
            output_dir,
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": dataset_name,
                "event": "node_prune",
                "original": len(nodes),
                "pruned": max_nodes,
                "reduction_pct": round(100 * (1 - max_nodes / len(nodes)), 1),
            },
        )
        nodes = random.sample(nodes, max_nodes)
        G = G.subgraph(nodes).copy()

    # Edge pruning
    if G.number_of_edges() > max_edges:
        edges = list(G.edges())
        _log_to_json(
            output_dir,
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": dataset_name,
                "event": "edge_prune",
                "original": len(edges),
                "pruned": max_edges,
                "reduction_pct": round(100 * (1 - max_edges / len(edges)), 1),
            },
        )
        edges_to_keep = random.sample(edges, max_edges)
        G = nx.Graph()
        G.add_edges_from(edges_to_keep)

    _log_to_json(
        output_dir,
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dataset_name,
            "event": "prune_complete",
            "final_nodes": G.number_of_nodes(),
            "final_edges": G.number_of_edges(),
            "original_nodes": original_stats["nodes"],
            "original_edges": original_stats["edges"],
        },
    )
    return G


def run_evaluation(
    dataset_name: str,
    G: nx.Graph,
    output_dir: str = "results",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run evaluation with JSON logging
    """
    # Initialize seeds
    random.seed(seed)
    np.random.seed(seed)

    # Start log
    _log_to_json(
        output_dir,
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dataset_name,
            "event": "evaluation_start",
        },
    )

    # Prune graph
    G = _prune_graph(G, output_dir, dataset_name, 5000, 15000)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    csv_dir = os.path.join(output_dir, "csvs")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # Run evaluation with seed
    predictor = EnhancedLinkPredictor(G, seed=seed)
    results = predictor.edge_recovery_evaluation(
        edge_removal_percentages=EVALUATION_CONFIG["edge_removal_percentages"],
        num_runs=EVALUATION_CONFIG["num_runs"],
    )

    # Save results
    results_path = os.path.join(csv_dir, f"{dataset_name}_edge_recovery.csv")
    results.to_csv(results_path, index=False)

    plt.figure(figsize=VIZ_CONFIG["figure_size"])
    sns.lineplot(
        data=results[results["metric"] == "true_positive_rate"],
        x="x_percent",
        y="mean",
        hue="algorithm",
        style="algorithm",
        markers=True,
        dashes=False,
    )
    for algo in results["algorithm"].unique():
        algo_data = results[
            (results["algorithm"] == algo) & (results["metric"] == "true_positive_rate")
        ]
        plt.fill_between(
            algo_data["x_percent"],
            algo_data["mean"] - algo_data["std"],
            algo_data["mean"] + algo_data["std"],
            alpha=VIZ_CONFIG["alpha"],
        )
    plt.title(f"Dataset - {dataset_name}")
    plt.xlabel("Edge Removal Percentage (%)")
    plt.ylabel("True Positive Rate")
    plt.grid(True, linestyle="--")
    plt.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", title="Algorithm", borderaxespad=0
    )
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f"{dataset_name}_recovery_plot.png")
    plt.savefig(plot_path, dpi=VIZ_CONFIG["dpi"], bbox_inches="tight")
    plt.close()

    # Plot confusion matrices for each x_percent and algorithm
    for x_percent in EVALUATION_CONFIG["edge_removal_percentages"]:
        for algo in results["algorithm"].unique():
            algo_data = results[
                (results["algorithm"] == algo) & (results["x_percent"] == x_percent)
            ]
            tp = algo_data[algo_data["metric"] == "true_positives"]["mean"].values[0]
            fp = algo_data[algo_data["metric"] == "false_positives"]["mean"].values[0]
            fn = algo_data[algo_data["metric"] == "false_negatives"]["mean"].values[0]
            tn = algo_data[algo_data["metric"] == "true_negatives"]["mean"].values[0]

            # Calculate percentages
            total = tp + fp + fn + tn
            tp_pct = 100 * tp / total
            fp_pct = 100 * fp / total
            fn_pct = 100 * fn / total
            tn_pct = 100 * tn / total

            # Create percentage confusion matrix
            confusion_matrix = np.array([[tp_pct, fp_pct], [fn_pct, tn_pct]])

            # Plot heatmap
            plt.figure(figsize=VIZ_CONFIG["figure_size"])
            sns.heatmap(
                confusion_matrix,
                annot=True,
                fmt=".1f",
                annot_kws={"size": 12},
                cmap="Blues",
                xticklabels=["Predicted Positive", "Predicted Negative"],
                yticklabels=["Actual Positive", "Actual Negative"],
            )
            plt.title(
                f"Confusion Matrix (% of total) - {algo.upper()} ({x_percent}% edges dropped)"
            )
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            # Save heatmap
            algo_plots_dir = os.path.join(plots_dir, dataset_name, algo)
            os.makedirs(algo_plots_dir, exist_ok=True)
            plt.savefig(
                os.path.join(algo_plots_dir, f"confusion_matrix_{x_percent}.png"),
                dpi=VIZ_CONFIG["dpi"],
                bbox_inches="tight",
            )
            plt.close()

    # Success log
    _log_to_json(
        output_dir,
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dataset_name,
            "event": "evaluation_complete",
            "result_files": {
                "data": os.path.relpath(results_path, output_dir),
                "plot": os.path.relpath(plot_path, output_dir),
            },
        },
    )
    return results
