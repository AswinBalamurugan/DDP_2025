from .imports import *


def _compute_katz(A):
    """Katz Index Computation using matrix inverse formula"""
    I = np.eye(A.shape[0])
    max_eigenvalue = max(np.linalg.eigvals(A))
    beta = np.real(0.8 / max_eigenvalue)
    matrix = I - beta * A
    inverse_matrix = np.linalg.inv(matrix)
    return inverse_matrix - I


def katz_score(node_index, katz_matrix, u, v):
    """Get precomputed Katz score using matrix inverse formula"""
    return katz_matrix[node_index[u], node_index[v]]
