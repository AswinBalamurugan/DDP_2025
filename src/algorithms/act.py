from .imports import *


def compute_laplacian_pseudoinverse(G):
    L = nx.laplacian_matrix(G).astype(float)
    try:
        L_pinv = np.linalg.pinv(L.toarray())
        return L_pinv
    except np.linalg.LinAlgError:
        return None


def act_score(G, node_index, L_pinv, u, v):
    """Average Commute Time between nodes u and v using pseudoinverse of Laplacian"""

    if L_pinv is None:
        return 0.0

    u_idx = node_index[u]
    v_idx = node_index[v]

    act = 1 / (L_pinv[u_idx, u_idx] + L_pinv[v_idx, v_idx] - 2 * L_pinv[u_idx, v_idx])
    return act if act > 0 else 0.0
