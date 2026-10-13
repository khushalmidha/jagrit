"""
Standardized Evaluation Metrics for News Recommendation.

Computes exact ranking, classification concordance, and diversity metrics:
- nDCG@5, nDCG@10
- Mean Reciprocal Rank (MRR)
- Area Under ROC Curve (ROC-AUC)
- Precision-Recall AUC (PR-AUC)
- Cohen's Kappa (chance-corrected agreement)
- Intra-List Diversity (ILD)
"""

from typing import List, Optional
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, cohen_kappa_score


def compute_dcg_at_k(labels: np.ndarray, k: int) -> float:
    """Computes Discounted Cumulative Gain at rank K."""
    labels_k = labels[:k]
    if len(labels_k) == 0:
        return 0.0
    discounts = np.log2(np.arange(len(labels_k)) + 2.0)
    return float(np.sum(labels_k / discounts))


def compute_ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank K."""
    order = np.argsort(y_score)[::-1]
    sorted_labels = y_true[order]
    dcg = compute_dcg_at_k(sorted_labels, k)

    ideal_labels = np.sort(y_true)[::-1]
    idcg = compute_dcg_at_k(ideal_labels, k)

    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def compute_mrr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Mean Reciprocal Rank (MRR) of first clicked article."""
    order = np.argsort(y_score)[::-1]
    sorted_labels = y_true[order]
    for rank, label in enumerate(sorted_labels, start=1):
        if label > 0:
            return float(1.0 / rank)
    return 0.0


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Computes Area Under ROC Curve. Returns None if single class."""
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Computes Precision-Recall AUC (Average Precision)."""
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return None


def compute_cohen_kappa(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """
    Computes Cohen's Kappa measuring chance-corrected concordance between
    the top-1 recommended item and actual user click labels.
    """
    if len(np.unique(y_true)) < 2:
        return None
    try:
        y_pred = np.zeros_like(y_true)
        y_pred[np.argmax(y_score)] = 1
        return float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        return None


def compute_intra_list_diversity(
    categories: List[str],
    top_k: int = 10,
) -> float:
    """
    Computes Intra-List Diversity (ILD) across top-K recommended articles
    as mean pairwise category distance: dist(c_i, c_j) = 1.0 if c_i != c_j else 0.0.
    """
    cats = categories[:top_k]
    n = len(cats)
    if n <= 1:
        return 0.0
    pairs = 0
    diff_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            if cats[i] != cats[j]:
                diff_count += 1
    return float(diff_count / max(1, pairs))
