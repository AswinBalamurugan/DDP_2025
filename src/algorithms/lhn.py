from .imports import *


def leicht_holme_newman(G, degrees, u, v):
    """Leicht-Holme-Newman Index"""
    cn = list(nx.common_neighbors(G, u, v))
    deg_u = degrees[u]
    deg_v = degrees[v]
    return len(cn) / (deg_u * deg_v) if (deg_u and deg_v) else 0.0
