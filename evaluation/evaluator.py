"""
Evaluator for image retrieval models.

This module handles:
- Computing embeddings for query and gallery sets
- Computing retrieval metrics
- Saving and visualizing results
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List
from tqdm import tqdm
import os
import json

from .metrics import compute_metrics, print_metrics


class RetrievalEvaluator:
    """
    Evaluator for image retrieval.
    
    Evaluation Process:
    -------------------
    1. Extract embeddings for all images (queries and gallery)
    2. Compute similarity between queries and gallery
    3. Rank gallery images for each query
    4. Compute metrics (mAP, Recall@K, etc.)
    5. Save results
    
    Memory Efficient:
    -----------------
    For large datasets, we process in batches to avoid OOM.
    Embeddings are computed once and cached.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        k_values: List[int] = [1, 5, 10]
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Embedding network
            device: Device to run on
            k_values: K values for Recall@K metrics
        """
        self.model = model
        self.device = device
        self.k_values = k_values
    
    @torch.no_grad()
    def extract_embeddings(
        self,
        dataloader,
        desc: str = "Extracting embeddings"
    ) -> tuple:
        """
        Extract embeddings for all images in dataloader.
        
        Args:
            dataloader: DataLoader for images
            desc: Description for progress bar
            
        Returns:
            Tuple of (embeddings, labels, indices)
            
        Why batch processing:
        ---------------------
        - Can't fit all images in GPU memory at once
        - Process in batches, accumulate results
        - More memory efficient
        """
        self.model.eval()
        
        all_embeddings = []
        all_labels = []
        all_indices = []
        
        pbar = tqdm(dataloader, desc=desc)
        
        for images, labels, indices in pbar:
            images = images.to(self.device)
            
            # Forward pass
            embeddings = self.model(images)
            
            # Move to CPU and store
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
            all_indices.append(indices)
        
        # Concatenate all batches
        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        indices = torch.cat(all_indices, dim=0).numpy()
        
        return embeddings, labels, indices
    
    def evaluate(
        self,
        query_loader,
        gallery_loader,
        save_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Args:
            query_loader: DataLoader for query images
            gallery_loader: DataLoader for gallery images
            save_dir: Directory to save results (optional)
            
        Returns:
            Dictionary of metrics
            
        Evaluation Steps:
        -----------------
        1. Extract query embeddings
        2. Extract gallery embeddings
        3. Compute metrics
        4. Save results (if save_dir provided)
        """
        # Extract embeddings
        print("\nEvaluating model...")
        query_embeddings, query_labels, query_indices = self.extract_embeddings(
            query_loader, desc="Query embeddings"
        )
        gallery_embeddings, gallery_labels, gallery_indices = self.extract_embeddings(
            gallery_loader, desc="Gallery embeddings"
        )
        
        print(f"Query set: {len(query_embeddings)} images")
        print(f"Gallery set: {len(gallery_embeddings)} images")
        
        # Compute metrics
        metrics = compute_metrics(
            query_embeddings,
            gallery_embeddings,
            query_labels,
            gallery_labels,
            k_values=self.k_values
        )
        
        # Print metrics
        print_metrics(metrics, title="Retrieval Performance")
        
        # Save results
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save metrics
            metrics_file = os.path.join(save_dir, 'metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to {metrics_file}")
            
            # Save embeddings
            embeddings_file = os.path.join(save_dir, 'embeddings.npz')
            np.savez(
                embeddings_file,
                query_embeddings=query_embeddings,
                query_labels=query_labels,
                query_indices=query_indices,
                gallery_embeddings=gallery_embeddings,
                gallery_labels=gallery_labels,
                gallery_indices=gallery_indices
            )
            print(f"Embeddings saved to {embeddings_file}")
        
        return metrics
    
    def compute_similarity_matrix(
        self,
        query_embeddings: np.ndarray,
        gallery_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity matrix between queries and gallery.
        
        Args:
            query_embeddings: Query embeddings (Q, D)
            gallery_embeddings: Gallery embeddings (G, D)
            
        Returns:
            Similarity matrix (Q, G)
            
        Uses cosine similarity (assumes normalized embeddings).
        """
        # Normalize
        query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        gallery_norm = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        similarities = np.dot(query_norm, gallery_norm.T)
        
        return similarities
    
    def get_top_k_retrievals(
        self,
        query_idx: int,
        query_embeddings: np.ndarray,
        gallery_embeddings: np.ndarray,
        k: int = 10
    ) -> np.ndarray:
        """
        Get top-K retrieved gallery images for a query.
        
        Args:
            query_idx: Index of query
            query_embeddings: All query embeddings
            gallery_embeddings: All gallery embeddings
            k: Number of top results
            
        Returns:
            Indices of top-K gallery images
            
        Useful for:
        -----------
        - Visualizing retrieval results
        - Debugging model behavior
        - Qualitative analysis
        """
        # Compute similarities for this query
        query_emb = query_embeddings[query_idx:query_idx+1]
        similarities = self.compute_similarity_matrix(query_emb, gallery_embeddings)[0]
        
        # Get top-K indices
        top_k_indices = np.argsort(-similarities)[:k]
        
        return top_k_indices

