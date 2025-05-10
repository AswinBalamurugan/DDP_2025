from .imports import *


def _init_hits_scores(A, max_iter=100, tol=1e-6):
    """HITS Algorithm Implementation"""
    hub = np.ones(A.shape[0])
    auth = np.ones(A.shape[0])

    for _ in range(max_iter):
        auth_new = A.T.dot(hub)
        hub_new = A.dot(auth)

        auth_norm = np.linalg.norm(auth_new)
        hub_norm = np.linalg.norm(hub_new)

        if auth_norm == 0 or hub_norm == 0:
            break

        auth_new = auth_new / auth_norm
        hub_new = hub_new / hub_norm

        # Check for convergence using tol
        auth_diff = np.linalg.norm(auth_new - auth)
        hub_diff = np.linalg.norm(hub_new - hub)
        if auth_diff < tol and hub_diff < tol:
            break

        auth = auth_new
        hub = hub_new

    return hub, auth


def hits_index(G, node_index, hub_scores, auth_scores, u, v):
    """HITS-based similarity score"""
    return hub_scores[node_index[u]] * auth_scores[node_index[v]]
