# Standard library imports
import os
import random
import pickle
import gzip
import shutil
from functools import partial
from typing import Dict, List, Tuple, Union

# Scientific computing imports
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import inv as sparse_inv

# Network/graph imports
import networkx as nx
from networkx.classes.graph import Graph

# Progress tracking
from tqdm import tqdm

# Plots
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import warnings

warnings.filterwarnings("ignore")
plt.style.use(["ieee", "no-latex"])
sns.set_palette("colorblind")
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["legend.title_fontsize"] = 14
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["figure.titlesize"] = 20
sns.set_style("dark")
