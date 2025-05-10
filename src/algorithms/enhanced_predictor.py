from .imports import *
from src.algorithms.lhn import leicht_holme_newman
from src.algorithms.hits import _init_hits_scores, hits_index
from src.algorithms.katz import _compute_katz, katz_score
from src.algorithms.act import act_score, compute_laplacian_pseudoinverse


class EnhancedLinkPredictor:
    """
    Enhanced link prediction class for graph analysis.
    """

    def __init__(self, G, seed: int = 42):
        """Initialize with a graph and seed."""
        if not isinstance(G, nx.Graph):
            raise ValueError("Input must be a NetworkX graph")
        self.G = G
        self.seed = seed
        self.nodes = list(G.nodes())
        self.node_index = {n: i for i, n in enumerate(self.nodes)}
        self.A = nx.adjacency_matrix(G).astype(np.float64)
        self.hub_scores, self.auth_scores = _init_hits_scores(self.A.toarray())
        self.katz_matrix = _compute_katz(self.A.toarray())
        self.L_pinv = compute_laplacian_pseudoinverse(G)
        self.degrees = dict(G.degree())
        self.non_edges = list(nx.non_edges(G))

    # ================== ALGORITHM IMPLEMENTATIONS ==================
    def lhn(self, u, v):
        """Leicht-Holme-Newman Index"""
        return leicht_holme_newman(self.G, self.degrees, u, v)

    def ra(self, u, v):
        """Resource Allocation Index"""
        cn = list(nx.common_neighbors(self.G, u, v))
        return sum(1 / (self.degrees[z] + 1e-9) for z in cn)

    def hits(self, u, v):
        """HITS-based similarity score"""
        return hits_index(
            self.G, self.node_index, self.hub_scores, self.auth_scores, u, v
        )

    def katz(self, u, v):
        """Katz Index"""
        return katz_score(self.node_index, self.katz_matrix, u, v)

    def act(self, u, v):
        """Average Commute Time"""
        return act_score(self.G, self.node_index, self.L_pinv, u, v)

    # ================== EVALUATION FRAMEWORK ==================
    def edge_recovery_evaluation(
        self, edge_removal_percentages=[10, 20, 30, 40], num_runs=100
    ):
        """Sequential evaluation framework with comprehensive metrics."""
        metrics = []
        algorithms = ["lhn", "ra", "hits", "katz", "act"]
        pred_metrics = [
            "true_positives",
            "false_positives",
            "false_negatives",
            "true_negatives",
            "true_positive_rate",
        ]

        # Precompute edges to remove for each percentage
        edges = list(self.G.edges())
        edges_to_remove = {
            x: random.sample(edges, k=int(len(edges) * x / 100))
            for x in edge_removal_percentages
        }

        with tqdm(edge_removal_percentages, desc="Edge Removal %") as pbar_x:
            for x in pbar_x:
                pbar_x.set_postfix_str(f"Current: {x}%")
                trial_results = {
                    alg: {rate: [] for rate in pred_metrics} for alg in algorithms
                }

                with tqdm(range(num_runs), desc="Trials", leave=False) as pbar_runs:
                    for run_id in pbar_runs:
                        trial = self._single_trial(
                            edges_to_remove[x], run_id, edges, self.non_edges
                        )
                        for alg in algorithms:
                            for rate in pred_metrics:
                                trial_results[alg][rate].append(trial[alg][rate])

                # Calculate statistics for each metric
                for alg in algorithms:
                    for rate in pred_metrics:
                        values = trial_results[alg][rate]
                        metrics.append(
                            {
                                "algorithm": alg,
                                "metric": rate,
                                "x_percent": x,
                                "mean": np.mean(values),
                                "std": np.std(values),
                                "min": np.min(values),
                                "max": np.max(values),
                                "q1": np.quantile(values, 0.25),
                                "q3": np.quantile(values, 0.75),
                            }
                        )

        return pd.DataFrame(metrics)

    def _single_trial(self, edges_to_remove, run_id, edges, non_edges):
        """Single trial execution with comprehensive metrics."""
        if run_id is not None:
            random.seed(self.seed + run_id)

        G_train = self.G.copy()
        G_train.remove_edges_from(edges_to_remove)
        train_predictor = EnhancedLinkPredictor(G_train, seed=self.seed + run_id)

        # To maintain original graph dynamics, use same number of non-edges as removed edges
        num_non_edges = len(edges_to_remove)
        sampled_non_edges = random.sample(non_edges, num_non_edges) if non_edges else []
        all_candidates = edges_to_remove + sampled_non_edges
        random.shuffle(all_candidates)

        # Pre-allocate arrays for scores
        n = len(all_candidates)
        scores = {
            "lhn": np.empty(n, dtype=float),
            "ra": np.empty(n, dtype=float),
            "hits": np.empty(n, dtype=float),
            "katz": np.empty(n, dtype=float),
            "act": np.empty(n, dtype=float),
        }

        # Compute scores in batches
        for i, (u, v) in enumerate(all_candidates):
            scores["lhn"][i] = np.real(train_predictor.lhn(u, v))
            scores["ra"][i] = np.real(train_predictor.ra(u, v))
            scores["hits"][i] = np.real(train_predictor.hits(u, v))
            scores["katz"][i] = np.real(train_predictor.katz(u, v))
            scores["act"][i] = np.real(train_predictor.act(u, v))

        metrics = {}
        k = len(edges_to_remove)
        removed_set = {tuple(sorted(e)) for e in edges_to_remove}
        non_edge_set = {tuple(sorted(e)) for e in sampled_non_edges}

        for alg in scores:
            ranked = sorted(zip(all_candidates, scores[alg]), key=lambda x: -x[1])
            top_k = {tuple(sorted(edge)) for edge, _ in ranked[:k]}

            # True positives (recovered edges)
            tp = len(top_k & removed_set)
            # False positives (non-edges in top k)
            fp = len(top_k & non_edge_set)
            # False negatives (edges not in top k)
            fn = len(removed_set - top_k)
            # True negatives (non-edges not in top k)
            tn = len(non_edge_set - top_k)

            # Calculate TPR (True Positive Rate)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            metrics[alg] = {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "true_positive_rate": tpr,
            }

        return metrics
