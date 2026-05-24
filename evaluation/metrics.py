"""
Evaluation metrics for image retrieval.

This module implements standard retrieval metrics:
- Mean Average Precision (mAP)
- Recall@K
- Precision@K
- NDCG@K

These are used to evaluate model performance on test sets.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


def compute_average_precision(
    retrieved_indices: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    """
    Compute Average Precision for a single query.
    
    Args:
        retrieved_indices: Indices of retrieved items (sorted by score)
        ground_truth: Binary array indicating relevant items
        
    Returns:
        Average Precision score
    """
    # Get relevance of retrieved items
    retrieved_relevance = ground_truth[retrieved_indices]
    
    # Compute precision at each position where relevant item appears
    cumsum = np.cumsum(retrieved_relevance)
    positions = np.arange(1, len(retrieved_indices) + 1)
    precision_at_k = cumsum / positions
    
    # Only consider positions with relevant items
    precision_at_relevant = precision_at_k * retrieved_relevance
    
    # Average over number of relevant items
    num_relevant = np.sum(ground_truth)
    if num_relevant > 0:
        ap = np.sum(precision_at_relevant) / num_relevant
    else:
        ap = 0.0
    
    return ap


def compute_recall_at_k(
    retrieved_indices: np.ndarray,
    ground_truth: np.ndarray,
    k: int
) -> float:
    """
    Compute Recall@K for a single query.
    
    Args:
        retrieved_indices: Indices of retrieved items
        ground_truth: Binary array indicating relevant items
        k: Number of top results to consider
        
    Returns:
        Recall@K score
    """
    # Get top-K retrieved items
    top_k = retrieved_indices[:k]
    
    # Count how many are relevant
    num_relevant_retrieved = np.sum(ground_truth[top_k])
    
    # Total number of relevant items
    num_relevant_total = np.sum(ground_truth)
    
    if num_relevant_total > 0:
        recall = num_relevant_retrieved / num_relevant_total
    else:
        recall = 0.0
    
    return recall


def compute_precision_at_k(
    retrieved_indices: np.ndarray,
    ground_truth: np.ndarray,
    k: int
) -> float:
    """
    Compute Precision@K for a single query.
    
    Args:
        retrieved_indices: Indices of retrieved items
        ground_truth: Binary array indicating relevant items
        k: Number of top results to consider
        
    Returns:
        Precision@K score
    """
    # Get top-K retrieved items
    top_k = retrieved_indices[:k]
    
    # Count how many are relevant
    num_relevant = np.sum(ground_truth[top_k])
    
    # Precision = relevant / k
    precision = num_relevant / k
    
    return precision


def compute_ndcg_at_k(
    retrieved_indices: np.ndarray,
    ground_truth: np.ndarray,
    k: int
) -> float:
    """
    Compute NDCG@K for a single query.
    
    Args:
        retrieved_indices: Indices of retrieved items
        ground_truth: Binary array indicating relevant items
        k: Number of top results to consider
        
    Returns:
        NDCG@K score
    """
    # Get top-K retrieved items
    top_k = retrieved_indices[:k]
    retrieved_relevance = ground_truth[top_k]
    
    # Compute DCG
    positions = np.arange(1, len(top_k) + 1)
    discount = 1.0 / np.log2(positions + 1)
    dcg = np.sum(retrieved_relevance * discount)
    
    # Compute IDCG (ideal DCG)
    num_relevant = min(np.sum(ground_truth), k)
    ideal_relevance = np.ones(num_relevant)
    ideal_positions = np.arange(1, num_relevant + 1)
    ideal_discount = 1.0 / np.log2(ideal_positions + 1)
    idcg = np.sum(ideal_relevance * ideal_discount)
    
    # Compute NDCG
    if idcg > 0:
        ndcg = dcg / idcg
    else:
        ndcg = 0.0
    
    return ndcg


def compute_metrics(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute all retrieval metrics.
    
    Args:
        query_embeddings: Query embeddings (Q, D)
        gallery_embeddings: Gallery embeddings (G, D)
        query_labels: Query labels (Q,)
        gallery_labels: Gallery labels (G,)
        k_values: K values for Recall@K and other K-based metrics
        
    Returns:
        Dictionary of metrics
        
    Process:
    --------
    1. Compute similarity matrix (queries × gallery)
    2. For each query:
       a. Sort gallery by similarity
       b. Determine which are relevant (same label)
       c. Compute metrics (mAP, Recall@K, etc.)
    3. Average across all queries
    """
    num_queries = len(query_embeddings)
    
    # Normalize embeddings for cosine similarity
    query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
    gallery_norm = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity matrix (cosine similarity)
    similarities = np.dot(query_norm, gallery_norm.T)
    
    # Initialize metric accumulators
    ap_scores = []
    recall_at_k = {k: [] for k in k_values}
    precision_at_k = {k: [] for k in k_values}
    ndcg_at_k = {k: [] for k in k_values}
    
    # Compute metrics for each query
    for i in range(num_queries):
        # Get similarities for this query
        query_sims = similarities[i]
        
        # Sort gallery by similarity (descending)
        retrieved_indices = np.argsort(-query_sims)
        
        # Determine relevant items (same label as query)
        ground_truth = (gallery_labels == query_labels[i]).astype(float)
        
        # Skip if no relevant items
        if np.sum(ground_truth) == 0:
            continue
        
        # Compute Average Precision
        ap = compute_average_precision(retrieved_indices, ground_truth)
        ap_scores.append(ap)
        
        # Compute Recall@K, Precision@K, NDCG@K
        for k in k_values:
            recall = compute_recall_at_k(retrieved_indices, ground_truth, k)
            precision = compute_precision_at_k(retrieved_indices, ground_truth, k)
            ndcg = compute_ndcg_at_k(retrieved_indices, ground_truth, k)
            
            recall_at_k[k].append(recall)
            precision_at_k[k].append(precision)
            ndcg_at_k[k].append(ndcg)
    
    # Aggregate metrics
    metrics = {
        'mAP': np.mean(ap_scores) if ap_scores else 0.0
    }
    
    for k in k_values:
        metrics[f'Recall@{k}'] = np.mean(recall_at_k[k]) if recall_at_k[k] else 0.0
        metrics[f'Precision@{k}'] = np.mean(precision_at_k[k]) if precision_at_k[k] else 0.0
        metrics[f'NDCG@{k}'] = np.mean(ndcg_at_k[k]) if ndcg_at_k[k] else 0.0
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics
        title: Title to display
    """
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print(f"{'=' * 50}")
    
    for metric_name, value in metrics.items():
        print(f"  {metric_name:.<30} {value:.4f}")
    
    print(f"{'=' * 50}\n")

