"""
Configuration parameters for the link prediction project
"""

DATASET_URLS = {
    "Bitcoin-Alpha": "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz",
    "Bio-Proteins": "https://nrvis.com/download/data/bio/bio-CE-CX.zip",
    "HepTh-Collab": "https://snap.stanford.edu/data/cit-HepTh.txt.gz",
    "Transport": "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat",
    "Enron-Email": "https://snap.stanford.edu/data/email-Enron.txt.gz",
    "EU-Email": "https://snap.stanford.edu/data/email-Eu-core.txt.gz",
    "Bitcoin-OTC": "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz",
    "Reddit-Hyperlinks": "https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv",
}

# Evaluation parameters
EVALUATION_CONFIG = {
    "edge_removal_percentages": [10, 25, 50, 75],
    "num_runs": 100,
}

# Visualization parameters
VIZ_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "alpha": 0.2,
}

# Dataset names
DATASET_NAMES = [
    "Bitcoin-Alpha",
    "Bio-Proteins",
    "HepTh-Collab",
    "Transport",
    "Enron-Email",
    "EU-Email",
    "Bitcoin-OTC",
    "Reddit-Hyperlinks",
]
