"""
Reward functions for RL-based image retrieval.

This is the CORE INNOVATION of the paper:
Instead of optimizing proxy losses (triplet, contrastive, etc.),
we directly optimize the task metric using reinforcement learning.

Key Idea:
---------
Traditional: Minimize loss → Hope this improves retrieval
RL Approach: Maximize retrieval metric directly!

Reward = Task Metric (mAP, Recall@K, NDCG, etc.)

This aligns training objective with evaluation metric,
leading to better performance on the actual task.

Process:
--------
1. Forward pass: Get embeddings for query and gallery
2. Compute similarities: Compare query to all gallery images
3. Rank gallery images by similarity
4. Compute reward: Evaluate ranking quality (mAP, Recall, etc.)
5. Backprop through policy: Use REINFORCE or similar
6. Update model to maximize expected reward

Challenges:
-----------
- Ranking is non-differentiable → Use policy gradients
- High variance in gradients → Use baselines and variance reduction
- Sparse rewards → Use intermediate rewards or reward shaping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional


def compute_average_precision(
    rankings: torch.Tensor,
    relevance: torch.Tensor
) -> torch.Tensor:
    """
    Compute Average Precision (AP) for a single query.
    
    Average Precision:
    ------------------
    AP = (1/R) * sum_{k=1}^{N} P(k) * rel(k)
    
    Where:
    - R: Total number of relevant items
    - P(k): Precision at rank k
    - rel(k): Relevance of item at rank k (1 if relevant, 0 otherwise)
    
    Why AP is a good metric:
    ------------------------
    - Considers both precision and recall
    - Rewards ranking relevant items higher
    - Commonly used in retrieval evaluation
    - Well-suited for ranking tasks
    
    Args:
        rankings: Indices of ranked items (N,) - sorted by similarity
        relevance: Binary relevance labels (N,) - 1 if relevant, 0 otherwise
        
    Returns:
        Average Precision score
        
    Example:
    --------
    Rankings: [2, 5, 1, 3, 4] (image IDs sorted by similarity)
    Relevance: [0, 1, 1, 0, 1] (which images are relevant)
    Ranked relevance: [1, 1, 0, 0, 1] (reorder by rankings)
    
    Precisions at relevant positions:
    - P@1 = 1/1 = 1.0 (1 relevant in top 1)
    - P@2 = 2/2 = 1.0 (2 relevant in top 2)
    - P@5 = 3/5 = 0.6 (3 relevant in top 5)
    
    AP = (1/3) * (1.0 + 1.0 + 0.6) = 0.867
    """
    # Reorder relevance by rankings
    sorted_relevance = relevance[rankings]
    
    # Compute precision at each position where relevant item is found
    # Cumsum gives number of relevant items seen so far
    cumsum_relevance = torch.cumsum(sorted_relevance, dim=0)
    
    # Precision at position k = (relevant items up to k) / k
    positions = torch.arange(1, len(rankings) + 1, device=rankings.device, dtype=torch.float32)
    precision_at_k = cumsum_relevance.float() / positions
    
    # Only consider positions where item is relevant
    precision_at_relevant = precision_at_k * sorted_relevance.float()
    
    # Average over number of relevant items
    num_relevant = sorted_relevance.sum()
    if num_relevant > 0:
        ap = precision_at_relevant.sum() / num_relevant
    else:
        ap = torch.tensor(0.0, device=rankings.device)
    
    return ap


def compute_per_query_ap(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    distance_metric: str = 'cosine'
) -> torch.Tensor:
    """Return a tensor of per-query Average Precision scores (Q,)."""
    num_queries = query_embeddings.size(0)

    if distance_metric == 'cosine':
        query_norm = F.normalize(query_embeddings, p=2, dim=1)
        gallery_norm = F.normalize(gallery_embeddings, p=2, dim=1)
        similarities = torch.mm(query_norm, gallery_norm.t())
    elif distance_metric == 'euclidean':
        similarities = -torch.cdist(query_embeddings, gallery_embeddings, p=2)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    ap_scores = []
    for i in range(num_queries):
        rankings = torch.argsort(similarities[i], descending=True)
        relevance = (gallery_labels == query_labels[i]).float()
        ap_scores.append(compute_average_precision(rankings, relevance))

    return torch.stack(ap_scores)


def compute_map(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    distance_metric: str = 'cosine'
) -> torch.Tensor:
    """Compute Mean Average Precision (scalar)."""
    return compute_per_query_ap(
        query_embeddings, gallery_embeddings,
        query_labels, gallery_labels, distance_metric
    ).mean()


def compute_recall_at_k(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    k: int = 1,
    distance_metric: str = 'cosine'
) -> torch.Tensor:
    """
    Compute Recall@K.
    
    Recall@K:
    ---------
    For each query, check if at least one relevant item
    appears in the top-K retrieved results.
    
    Recall@K = (# queries with relevant item in top-K) / (# queries)
    
    Why Recall@K:
    -------------
    - Simpler than mAP
    - User-centric: "Did I find what I wanted in top K results?"
    - Different K values show different aspects:
      * Recall@1: Strictest, first result must be correct
      * Recall@5: More lenient, correct result in top 5
      * Recall@10: Even more lenient
    
    Common in:
    ----------
    - Face recognition (Recall@1)
    - Product search (Recall@10)
    - Visual search (Recall@5)
    
    Args:
        query_embeddings: Query embeddings (Q, D)
        gallery_embeddings: Gallery embeddings (G, D)
        query_labels: Query labels (Q,)
        gallery_labels: Gallery labels (G,)
        k: Number of top results to consider
        distance_metric: Distance metric to use
        
    Returns:
        Recall@K score
    """
    num_queries = query_embeddings.size(0)
    
    # Compute similarity matrix
    if distance_metric == 'cosine':
        query_norm = F.normalize(query_embeddings, p=2, dim=1)
        gallery_norm = F.normalize(gallery_embeddings, p=2, dim=1)
        similarities = torch.mm(query_norm, gallery_norm.t())
    elif distance_metric == 'euclidean':
        dist_sq = torch.cdist(query_embeddings, gallery_embeddings, p=2)
        similarities = -dist_sq
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    # Get top-K indices for each query (cap k to gallery size)
    # Shape: (Q, K)
    k_safe = min(k, similarities.size(1))
    top_k_indices = torch.topk(similarities, k=k_safe, dim=1)[1]
    
    # Check if any top-K result is relevant
    correct = 0
    for i in range(num_queries):
        top_k_labels = gallery_labels[top_k_indices[i]]
        query_label = query_labels[i]
        
        # Check if query label appears in top-K
        if (top_k_labels == query_label).any():
            correct += 1
    
    recall = correct / num_queries
    return torch.tensor(recall, device=query_embeddings.device)


def compute_ndcg_at_k(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    k: int = 10,
    distance_metric: str = 'cosine'
) -> torch.Tensor:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K).
    
    NDCG:
    -----
    Measures ranking quality with position-based discounting.
    Items ranked higher contribute more to the score.
    
    DCG@K = sum_{i=1}^{K} (2^{rel_i} - 1) / log_2(i + 1)
    NDCG@K = DCG@K / IDCG@K
    
    Where:
    - rel_i: Relevance of item at position i
    - IDCG: Ideal DCG (best possible ranking)
    
    Why NDCG:
    ---------
    - Position-aware: Ranking matters
    - Normalized: Always between 0 and 1
    - Flexible: Can handle graded relevance (not just binary)
    - Popular in ranking evaluation
    
    Args:
        query_embeddings: Query embeddings (Q, D)
        gallery_embeddings: Gallery embeddings (G, D)
        query_labels: Query labels (Q,)
        gallery_labels: Gallery labels (G,)
        k: Number of top results to consider
        distance_metric: Distance metric to use
        
    Returns:
        NDCG@K score
    """
    num_queries = query_embeddings.size(0)
    
    # Compute similarity matrix
    if distance_metric == 'cosine':
        query_norm = F.normalize(query_embeddings, p=2, dim=1)
        gallery_norm = F.normalize(gallery_embeddings, p=2, dim=1)
        similarities = torch.mm(query_norm, gallery_norm.t())
    elif distance_metric == 'euclidean':
        dist_sq = torch.cdist(query_embeddings, gallery_embeddings, p=2)
        similarities = -dist_sq
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    ndcg_scores = []
    
    # Precompute discount factors: 1/log2(i+1) for i=1..k
    positions = torch.arange(1, k + 1, device=query_embeddings.device, dtype=torch.float32)
    discount = 1.0 / torch.log2(positions + 1)
    
    for i in range(num_queries):
        # Get top-K indices
        top_k_indices = torch.topk(similarities[i], k=min(k, similarities.size(1)))[1]
        
        # Get relevance: 1 if same label, 0 otherwise
        top_k_relevance = (gallery_labels[top_k_indices] == query_labels[i]).float()
        
        # Compute DCG
        dcg = (top_k_relevance * discount[:len(top_k_indices)]).sum()
        
        # Compute IDCG (ideal DCG with perfect ranking)
        num_relevant = (gallery_labels == query_labels[i]).sum().item()
        ideal_relevance = torch.ones(min(num_relevant, k), device=query_embeddings.device)
        idcg = (ideal_relevance * discount[:len(ideal_relevance)]).sum()
        
        # Compute NDCG
        if idcg > 0:
            ndcg = dcg / idcg
        else:
            ndcg = torch.tensor(0.0, device=query_embeddings.device)
        
        ndcg_scores.append(ndcg)
    
    return torch.stack(ndcg_scores).mean()


class RetrievalReward(nn.Module):
    """
    Reward function for RL-based image retrieval training.
    
    This is the KEY COMPONENT that enables direct optimization
    of retrieval metrics using reinforcement learning.
    
    How it works:
    -------------
    1. Model produces embeddings (this is the "action")
    2. We evaluate retrieval performance using these embeddings
    3. Retrieval metric (mAP, Recall@K) is the reward
    4. Use policy gradients to maximize expected reward
    
    Why this is powerful:
    ---------------------
    - Direct optimization of the task metric
    - No mismatch between training and evaluation
    - Can optimize non-differentiable metrics
    - Proven to outperform proxy losses
    
    Challenges:
    -----------
    - High variance in gradients → Use variance reduction
    - Sparse rewards → Use reward shaping
    - Sample efficiency → Use experience replay (optional)
    """
    
    def __init__(
        self,
        metric: str = 'map',
        k_values: Optional[List[int]] = None,
        baseline_type: str = 'moving_average',
        baseline_momentum: float = 0.9,
        distance_metric: str = 'cosine'
    ):
        """
        Initialize reward function.
        
        Args:
            metric: Reward metric ('map', 'recall', 'ndcg', 'combined')
            k_values: K values for recall@k or ndcg@k
            baseline_type: Baseline for variance reduction
                          ('none', 'moving_average', 'learned')
            baseline_momentum: Momentum for moving average baseline
            distance_metric: Distance metric for similarity
            
        Baseline:
        ---------
        Policy gradient has high variance. We use a baseline to reduce it:
        
        Vanilla: ∇L = E[R * ∇log π]
        With baseline: ∇L = E[(R - b) * ∇log π]
        
        The baseline doesn't bias the gradient (E[b * ∇log π] = 0)
        but reduces variance if b ≈ E[R].
        
        Common baselines:
        - Moving average of past rewards
        - Value function (learned, as in Actor-Critic)
        - Constant (simple but less effective)
        """
        super(RetrievalReward, self).__init__()
        
        self.metric = metric
        self.k_values = k_values or [1, 5, 10]
        self.baseline_type = baseline_type
        self.baseline_momentum = baseline_momentum
        self.distance_metric = distance_metric
        
        # Initialize moving average baseline
        if baseline_type == 'moving_average':
            self.register_buffer('baseline', torch.tensor(0.0))
            self.register_buffer('baseline_initialized', torch.tensor(False))
        
        print(f"RetrievalReward: metric={metric}, baseline={baseline_type}")
    
    def compute_reward(
        self,
        query_embeddings: torch.Tensor,
        gallery_embeddings: torch.Tensor,
        query_labels: torch.Tensor,
        gallery_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reward based on retrieval performance.
        
        Args:
            query_embeddings: Query embeddings (Q, D)
            gallery_embeddings: Gallery embeddings (G, D)
            query_labels: Query labels (Q,)
            gallery_labels: Gallery labels (G,)
            
        Returns:
            Reward value (scalar)
        """
        if self.metric == 'map':
            reward = compute_map(
                query_embeddings, gallery_embeddings,
                query_labels, gallery_labels,
                self.distance_metric
            )
        
        elif self.metric == 'recall':
            # Average over multiple K values (skip k > gallery size)
            gallery_size = gallery_embeddings.size(0)
            recalls = []
            for k in self.k_values:
                if k > gallery_size:
                    continue
                recall = compute_recall_at_k(
                    query_embeddings, gallery_embeddings,
                    query_labels, gallery_labels,
                    k, self.distance_metric
                )
                recalls.append(recall)
            reward = torch.stack(recalls).mean() if recalls else torch.tensor(0.0, device=query_embeddings.device)
        
        elif self.metric == 'ndcg':
            # Average over multiple K values (skip k > gallery size)
            gallery_size = gallery_embeddings.size(0)
            ndcgs = []
            for k in self.k_values:
                if k > gallery_size:
                    continue
                ndcg = compute_ndcg_at_k(
                    query_embeddings, gallery_embeddings,
                    query_labels, gallery_labels,
                    k, self.distance_metric
                )
                ndcgs.append(ndcg)
            reward = torch.stack(ndcgs).mean() if ndcgs else torch.tensor(0.0, device=query_embeddings.device)

        elif self.metric == 'combined':
            # 0.5 * mAP + 0.5 * mean(recall@k) for better balance across datasets
            map_r = compute_map(
                query_embeddings, gallery_embeddings,
                query_labels, gallery_labels,
                self.distance_metric
            )
            gallery_size = gallery_embeddings.size(0)
            recalls = []
            for k in self.k_values:
                if k > gallery_size:
                    continue
                r = compute_recall_at_k(
                    query_embeddings, gallery_embeddings,
                    query_labels, gallery_labels,
                    k, self.distance_metric
                )
                recalls.append(r)
            recall_r = torch.stack(recalls).mean() if recalls else torch.tensor(0.0, device=query_embeddings.device)
            reward = 0.5 * map_r + 0.5 * recall_r

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        return reward

    def compute_per_example_reward(
        self,
        query_embeddings: torch.Tensor,
        gallery_embeddings: torch.Tensor,
        query_labels: torch.Tensor,
        gallery_labels: torch.Tensor
    ) -> torch.Tensor:
        """Per-query reward tensor of shape (Q,)."""
        return compute_per_query_ap(
            query_embeddings, gallery_embeddings,
            query_labels, gallery_labels,
            self.distance_metric
        )

    def get_baseline(self) -> torch.Tensor:
        """
        Get current baseline value.
        
        Returns:
            Baseline value for variance reduction
        """
        if self.baseline_type == 'none':
            return torch.tensor(0.0, device=self.baseline.device if hasattr(self, 'baseline') else 'cpu')
        
        elif self.baseline_type == 'moving_average':
            return self.baseline
        
        else:
            raise NotImplementedError(f"Baseline type {self.baseline_type} not implemented")
    
    def update_baseline(self, reward: torch.Tensor):
        """
        Update baseline with new reward.
        
        Args:
            reward: New reward value
            
        For moving average:
        baseline = momentum * baseline + (1 - momentum) * reward
        """
        if self.baseline_type == 'moving_average':
            if not self.baseline_initialized:
                # Initialize with first reward
                self.baseline.copy_(reward.detach())
                self.baseline_initialized.fill_(True)
            else:
                # Exponential moving average
                self.baseline.mul_(self.baseline_momentum).add_(
                    reward.detach(), alpha=1 - self.baseline_momentum
                )

