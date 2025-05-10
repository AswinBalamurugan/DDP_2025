import urllib.request
import zipfile
from collections import defaultdict
from src.algorithms.imports import *
from config.hyperparams import DATASET_URLS, DATASET_NAMES


def preprocess_graph(G, output_dir="processed_graphs"):
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    graph_name = getattr(G, "name", "unnamed_graph")

    original_path = os.path.join(output_dir, f"{graph_name}_original.pkl")
    with open(original_path, "wb") as f:
        pickle.dump(G, f)

    # Dictionary to store different versions
    graph_versions = {"original": G}

    return graph_versions


def create_graph_version(G, edge_removal_percentage, run_id, random_seed=None):
    seed = random_seed if random_seed is not None else run_id
    random.seed(seed)  # Ensure reproducibility for this run

    edges = list(G.edges())
    num_remove = int(len(edges) * edge_removal_percentage)
    removed_edges = random.sample(edges, num_remove)

    G_train = G.copy()
    G_train.remove_edges_from(removed_edges)
    return G_train, removed_edges


def load_graph(dataset_name, version="original", processed_dir="data"):
    """Load a graph (original or processed), skipping download if already exists."""
    if version == "original":
        if dataset_name not in DATASET_NAMES:
            raise ValueError(
                f"Dataset {dataset_name} not supported. Available datasets: {DATASET_NAMES}"
            )

        # Create data directory if it doesn't exist
        os.makedirs(processed_dir, exist_ok=True)

        # Load the graph based on the dataset name
        if dataset_name == "Bitcoin-Alpha":
            return _load_bitcoin_alpha(processed_dir)
        elif dataset_name == "Bio-Proteins":
            return _load_bio_proteins(processed_dir)
        elif dataset_name == "HepTh-Collab":
            return _load_hepth_collab(processed_dir)
        elif dataset_name == "Transport":
            return _load_transport(processed_dir)
        elif dataset_name == "Enron-Email":
            return _load_enron_email(processed_dir)
        elif dataset_name == "EU-Email":
            return _load_eu_email(processed_dir)
        elif dataset_name == "Bitcoin-OTC":
            return _load_bitcoin_otc(processed_dir)
        elif dataset_name == "Reddit-Hyperlinks":
            return _load_reddit_hyperlinks(processed_dir)
    else:
        raise NotImplementedError("Preprocessed graphs not yet supported")


def _load_bitcoin_alpha(processed_dir):
    """Load Bitcoin-Alpha dataset."""
    csv_path = os.path.join(processed_dir, "soc-sign-bitcoinalpha.csv")
    if not os.path.exists(csv_path):
        gz_path = os.path.join(processed_dir, "soc-sign-bitcoinalpha.csv.gz")
        urllib.request.urlretrieve(DATASET_URLS["Bitcoin-Alpha"], gz_path)
        with gzip.open(gz_path, "rb") as f_in:
            with open(csv_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    bitcoin_edges = pd.read_csv(
        csv_path,
        header=None,
        names=["source", "target", "weight", "time"],
    )
    G = nx.Graph()
    for _, row in bitcoin_edges.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["weight"])
    return G


def _load_bio_proteins(processed_dir):
    """Load Bio-Proteins dataset."""
    edges_path = os.path.join(processed_dir, "bio-CE-CX.edges")
    if not os.path.exists(edges_path):
        zip_path = os.path.join(processed_dir, "bio-CE-CX.zip")
        urllib.request.urlretrieve(DATASET_URLS["Bio-Proteins"], zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(processed_dir)
    bio_edges = pd.read_csv(
        edges_path,
        sep=" ",
        header=None,
        names=["source", "target", "weight"],
    )
    G = nx.Graph()
    for _, row in bio_edges.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["weight"])
    return G


def _load_hepth_collab(processed_dir):
    """Load HepTh-Collab dataset."""
    gz_path = os.path.join(processed_dir, "cit-HepTh.txt.gz")
    txt_path = os.path.join(processed_dir, "cit-HepTh.txt")
    if not os.path.exists(gz_path):
        urllib.request.urlretrieve(DATASET_URLS["HepTh-Collab"], gz_path)
    if not os.path.exists(txt_path):
        with gzip.open(gz_path, "rb") as f_in:
            with open(txt_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    G = nx.Graph()
    edge_counts = defaultdict(int)
    with open(txt_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                source = parts[0]
                target = parts[1]
                edge_counts[(source, target)] += 1
    for (source, target), count in edge_counts.items():
        G.add_edge(source, target, weight=count)
    return G


def _load_transport(processed_dir):
    """Load Transport dataset."""
    csv_path = os.path.join(processed_dir, "openflights.csv")
    if not os.path.exists(csv_path):
        urllib.request.urlretrieve(DATASET_URLS["Transport"], csv_path)
    flights = pd.read_csv(
        csv_path,
        header=None,
        names=[
            "airline",
            "airline_id",
            "source",
            "source_id",
            "dest",
            "dest_id",
            "codeshare",
            "stops",
            "equipment",
        ],
    )
    G = nx.Graph()
    route_counts = (
        flights.groupby(["source_id", "dest_id"]).size().reset_index(name="weight")
    )
    for _, row in route_counts.iterrows():
        G.add_edge(row["source_id"], row["dest_id"], weight=row["weight"])
    return G


def _load_enron_email(processed_dir):
    """Load Enron-Email dataset."""
    txt_path = os.path.join(processed_dir, "email-Enron.txt")
    if not os.path.exists(txt_path):
        gz_path = os.path.join(processed_dir, "email-Enron.txt.gz")
        urllib.request.urlretrieve(DATASET_URLS["Enron-Email"], gz_path)
        with gzip.open(gz_path, "rb") as f_in:
            with open(txt_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
    email_counts = defaultdict(int)
    with open(txt_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            src, dst = map(int, line.strip().split())
            email_counts[(src, dst)] += 1
    G = nx.Graph()
    for (src, dst), count in email_counts.items():
        G.add_edge(src, dst, weight=count)
    return G


def _load_eu_email(processed_dir):
    """Load EU-Email dataset."""
    gz_path = os.path.join(processed_dir, "email-Eu-core.txt.gz")
    txt_path = os.path.join(processed_dir, "email-Eu-core.txt")
    if not os.path.exists(gz_path):
        urllib.request.urlretrieve(DATASET_URLS["EU-Email"], gz_path)
        with gzip.open(gz_path, "rb") as f_in:
            with open(txt_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
    email_counts = defaultdict(int)
    with open(txt_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            src, dst = map(int, line.strip().split())
            email_counts[(src, dst)] += 1
    G = nx.Graph()
    for (src, dst), count in email_counts.items():
        G.add_edge(src, dst, weight=count)
    return G


def _load_bitcoin_otc(processed_dir):
    """Load Bitcoin-OTC dataset."""
    csv_path = os.path.join(processed_dir, "soc-sign-bitcoinotc.csv")
    if not os.path.exists(csv_path):
        gz_path = os.path.join(processed_dir, "soc-sign-bitcoinotc.csv.gz")
        urllib.request.urlretrieve(DATASET_URLS["Bitcoin-OTC"], gz_path)
        with gzip.open(gz_path, "rb") as f_in:
            with open(csv_path, "wb") as f_out:
                f_out.write(f_in.read())
        os.remove(gz_path)
    bitcoinotc_edges = pd.read_csv(
        csv_path,
        header=None,
        names=["source", "target", "weight", "time"],
    )
    G = nx.Graph()
    for _, row in bitcoinotc_edges.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["weight"])
    return G


def _load_reddit_hyperlinks(processed_dir):
    """Load Reddit Hyperlink Network"""
    tsv_path = os.path.join(processed_dir, "soc-redditHyperlinks-body.tsv")
    if not os.path.exists(tsv_path):
        urllib.request.urlretrieve(
            DATASET_URLS["Reddit-Hyperlinks"],
            tsv_path,
        )

    # Load the dataset with sentiment scores as weights
    reddit_df = pd.read_csv(tsv_path, sep="\t")
    G = nx.Graph()

    # Use sentiment scores as edge weights
    for _, row in reddit_df.iterrows():
        source = row["SOURCE_SUBREDDIT"]
        target = row["TARGET_SUBREDDIT"]
        sentiment = float(row["LINK_SENTIMENT"])
        weight = (
            1.0 + sentiment
            if sentiment > 0
            else 1.0 / (1.0 - sentiment) if sentiment < 0 else 1.0
        )
        G.add_edge(source, target, weight=weight)
    return G
