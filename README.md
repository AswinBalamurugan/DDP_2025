# Link Prediction Framework

A comprehensive framework for graph-based link prediction with multiple algorithms and evaluation methodologies. This repository implements state-of-the-art link prediction techniques and provides automated testing across multiple real-world network datasets.

## Project Objective

The primary objective of this framework is to evaluate and compare the performance of various link prediction algorithms across diverse network datasets. By systematically removing edges and attempting to predict them, we can measure each algorithm's effectiveness at recovering missing connections in different types of networks.

This framework enables researchers and data scientists to:

- Test multiple link prediction algorithms on various graph datasets
- Compare algorithm performance using standardized metrics
- Visualize results through recovery plots and confusion matrices
- Process and manage large network datasets efficiently

## Implementation Details

### Supported Datasets

The framework supports evaluation on eight diverse real-world network datasets, sourced from publicly available repositories. Below are the datasets and their download URLs:

| Dataset Name          | Description                                        | Download URL                                                                              |
| --------------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **Bitcoin-Alpha**     | Trust network from the Bitcoin Alpha platform      | [Download](https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz)                   |
| **Bio-Proteins**      | Protein interaction network                        | [Download](https://nrvis.com/download/data/bio/bio-CE-CX.zip)                             |
| **HepTh-Collab**      | Collaboration network of HEP-TH researchers        | [Download](https://snap.stanford.edu/data/cit-HepTh.txt.gz)                               |
| **Transport**         | Air transportation network (OpenFlights)           | [Download](https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat) |
| **Enron-Email**       | Email communication network from Enron             | [Download](https://snap.stanford.edu/data/email-Enron.txt.gz)                             |
| **EU-Email**          | Email network from a European research institution | [Download](https://snap.stanford.edu/data/email-Eu-core.txt.gz)                           |
| **Bitcoin-OTC**       | Trust network from Bitcoin OTC platform            | [Download](https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz)                     |
| **Reddit-Hyperlinks** | Network of subreddit connections via hyperlinks    | [Download](https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv)                  |

## Directory Structure

```text
.
├── config
│   └── hyperparams.py           # Configuration parameters
├── data                         # Dataset storage directory
│   ├── bio-CE-CX.edges
│   ├── cit-HepTh.txt
│   ├── email-Enron.txt
│   └── ...                      # Other dataset files
├── main.py                      # Main execution script
├── results                      # Results directory
│   ├── csvs                     # Edge recovery CSVs for all datasets (contains one CSV per dataset)
│   │   └── ...                  # Dataset-specific edge recovery files
│   ├── final_results.json       # Compiled evaluation results
│   ├── plots                    # Visualization outputs
│   │   ├── {dataset_name}       # Dataset-specific directory (repeated for each dataset)
│   │   │   └── heatmaps         # Confusion matrices by algorithm
│   │   │       ├── act          # Average Commute Time results
│   │   │       ├── hits         # HITS algorithm results
│   │   │       ├── katz         # Katz index results
│   │   │       ├── lhn          # Leicht-Holme-Newman results
│   │   │       └── ra           # Resource Allocation results
│   │   └── {dataset_name}_recovery_plot.png  # Recovery plot for each dataset
│   ├── progress.json            # Progress tracking metadata
│   └── progress_log.json        # Detailed execution logs
└── src                          # Source code
    ├── algorithms               # Algorithm implementations
    │   ├── __init__.py
    │   ├── act.py               # Average Commute Time
    │   ├── enhanced_predictor.py # Main predictor class
    │   ├── hits.py              # HITS algorithm
    │   ├── imports.py           # Common imports
    │   ├── katz.py              # Katz index
    │   └── lhn.py               # Leicht-Holme-Newman
    ├── data                     # Data management
    │   ├── __init__.py
    │   └── graph_manager.py     # Graph loading and processing
    └── evaluation               # Evaluation framework
        ├── __init__.py
        └── evaluator.py         # Evaluation logic
```

## Code Flow

The pipeline is modular and follows these steps:

1. **Data Loading** (`graph_manager.py`)

   - Downloads and preprocesses datasets if not present.
   - Loads graphs in a standard format for downstream tasks.

2. **Link Prediction** (`enhanced_predictor.py`)

   - Initializes with a graph.
   - Implements all prediction algorithms.
   - Creates training graphs by removing edges.
   - Predicts missing links using various algorithms.

3. **Evaluation** (`evaluator.py`)

   - Implements edge recovery evaluation methodology.
   - Calculates metrics like true/false positive rates.
   - Generates confusion matrices and performance plots.
   - Saves results in CSV format and as visualizations.

4. **Execution Management** (`main.py`)
   - Parses command line arguments.
   - Orchestrates the evaluation process.
   - Handles logging and progress tracking.
   - Compiles final results.

## Algorithm Explanations

| Algorithm | Formula                                                                            | Intuition                                                                           |
| --------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **LHN**   | $$\frac{\text{common\_neighbours}(u,v)}{\text{degree}(u) \cdot \text{degree}(v)}$$ | Higher if nodes share many neighbours and have low degree.                          |
| **RA**    | $$\sum_{z \in \text{common\_neighbours}(u,v)} \frac{1}{\text{degree}(z)}$$         | Nodes share rare neighbours, making them more likely to connect.                    |
| **HITS**  | $$\text{hub\_score}(u) \times \text{authority\_score}(v)$$                         | Combines node roles as hubs (linking to others) and authorities (linked to).        |
| **Katz**  | $$(I - \beta A)^{-1} - I$$                                                         | Sums all paths between nodes, with shorter paths weighted more heavily.             |
| **ACT**   | $$\frac{1}{L^+[u,u] + L^+[v,v] - 2L^+[u,v]}$$                                      | Measures expected random walk commute time; lower values indicate nodes are closer. |

## Evaluation Framework

The framework evaluates link prediction algorithms using a systematic edge recovery methodology:

1. **Edge Removal**:

   - A percentage of edges (configurable via `EVALUATION_CONFIG["edge_removal_percentages"]`) is randomly removed from the original graph.
   - Default removal percentages: `[10, 20, 30, 40, 50]`.

2. **Training Graph Creation**:

   - The remaining edges form the training graph, which is used for link prediction.

3. **Prediction**:

   - Each algorithm ranks non-existing edges (including the removed ones) based on their likelihood of forming a link.

4. **Evaluation**:

   - The top-K predicted edges are compared against the removed edges to calculate recovery metrics.
   - Metrics include **True Positive Rate (TPR)**, **False Positive Rate (FPR)**, and **Accuracy**.
   - The process is repeated for `num_runs` (default: 20) to ensure statistical robustness.

5. **Visualization**:
   - Results are visualized using plots (e.g., recovery curves, confusion matrices) with parameters defined in `VIZ_CONFIG`.

### Confusion Matrix Interpretation

Confusion matrices are generated for each algorithm and edge removal percentage:

|                 | Predicted Positive  | Predicted Negative  |
| --------------- | ------------------- | ------------------- |
| Actual Positive | True Positive Rate  | False Negative Rate |
| Actual Negative | False Positive Rate | True Negative Rate  |

## Usage Instructions

### Installation

Clone the repository:

```code
git clone https://github.com/yourusername/link-prediction-framework.git
cd link-prediction-framework
```

Install dependencies:

```code
pip install -r requirements.txt
```

### Running Evaluations

**Run evaluation on all datasets:**

```code
python main.py
```

**Run evaluation on specific datasets:**

```code
python main.py --datasets Bitcoin-Alpha Bio-Proteins
```

**Specify custom output directory:**

```code
python main.py --output_dir custom_results
```

## Results

- **CSV files**: Detailed metrics for each algorithm, dataset, and edge removal percentage.
- **Recovery plots**: Show how well each algorithm recovers missing edges at different removal percentages.
- **Confusion matrices**: Visualise prediction accuracy for each algorithm.

## Extending the Framework

**Adding a new algorithm:**

- Add its implementation under `src/algorithms/`.
- Register it in `EnhancedLinkPredictor`.
- Update evaluation scripts if needed.

**Adding a new dataset:**

- Add its URL and name in `config/hyperparams.py`.
- Implement a loader in `graph_manager.py`.
