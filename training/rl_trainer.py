"""
Reinforcement Learning trainer for image retrieval.

This module implements policy gradient methods to directly optimize
retrieval metrics. This is the CORE INNOVATION of the paper.

Key Algorithms Implemented:
----------------------------
1. REINFORCE (Vanilla Policy Gradient)
2. REINFORCE with Baseline
3. Actor-Critic (optional, for future extension)

The RL Training Loop:
---------------------
1. Sample batch of queries and gallery
2. Forward pass: compute embeddings
3. Compute reward: evaluate retrieval performance
4. Compute policy gradient
5. Update model parameters
6. Update baseline (if used)

Why RL for Retrieval:
---------------------
Traditional approaches optimize proxy losses (triplet, contrastive)
and HOPE they improve retrieval metrics. RL directly optimizes
the metrics we care about!

Benefits:
- Direct optimization of task metric (mAP, Recall@K)
- No mismatch between training and evaluation
- Better final performance
- Can handle non-differentiable metrics

Challenges:
- High variance gradients → Use baselines
- Sample efficiency → Use experience replay
- Computational cost → Batch processing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Tuple
from tqdm import tqdm
import time

from losses.rl_rewards import RetrievalReward
from training.rl_estimators import multi_sample_policy_loss
from training.pl_policy import pl_policy_loss, softmax_control_loss, smoothap_control_loss, fastap_bank_loss
from utils.helpers import AverageMeter, format_time
from utils.logger import Logger


class EmbeddingMemoryBank:
    """FIFO queue of (embedding, label) pairs from recent batches.

    Used as additional gallery items when computing the RL reward so that
    the retrieval problem is hard enough to provide a meaningful gradient
    signal.  Without this, the gallery is just half of one mini-batch
    (~8 items) and recall / mAP saturates at ~1.0.
    """

    def __init__(self, capacity: int = 4096):
        self.capacity = capacity
        self.embeddings: torch.Tensor | None = None
        self.labels: torch.Tensor | None = None
        self._ptr = 0
        self._full = False

    def reset(self):
        self.embeddings = None
        self.labels = None
        self._ptr = 0
        self._full = False

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, labels: torch.Tensor):
        bs = embeddings.shape[0]
        if self.embeddings is None:
            self.embeddings = torch.zeros(
                self.capacity, embeddings.shape[1], device=embeddings.device
            )
            self.labels = torch.zeros(
                self.capacity, dtype=labels.dtype, device=labels.device
            )

        if self._ptr + bs <= self.capacity:
            self.embeddings[self._ptr : self._ptr + bs] = embeddings
            self.labels[self._ptr : self._ptr + bs] = labels
            self._ptr += bs
        else:
            tail = self.capacity - self._ptr
            self.embeddings[self._ptr :] = embeddings[:tail]
            self.labels[self._ptr :] = labels[:tail]
            head = bs - tail
            self.embeddings[:head] = embeddings[tail:]
            self.labels[:head] = labels[tail:]
            self._ptr = head
            self._full = True

        if self._ptr >= self.capacity:
            self._ptr = 0
            self._full = True

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._full:
            return self.embeddings, self.labels
        return self.embeddings[: self._ptr], self.labels[: self._ptr]

    def __len__(self) -> int:
        return self.capacity if self._full else self._ptr


class RLTrainer:
    """REINFORCE trainer aligned with 'Tuning CV Models with Task Rewards'.

    Key differences from vanilla REINFORCE on embeddings:
      - Per-example reward: each query gets its own AP and its own advantage.
      - Sample baseline: a second independent sample serves as baseline
        (adapts to per-batch class composition, like the paper's Algorithm 2).
      - Log-prob computed on the unit sphere (cosine distance after projection)
        rather than in raw Euclidean space.
      - BatchNorm frozen during RL to preserve pretrained statistics.
    """

    def __init__(
        self,
        model: nn.Module,
        reward_fn: RetrievalReward,
        optimizer: optim.Optimizer,
        logger: Logger,
        device: torch.device,
        reference_model: nn.Module = None,
        gradient_clip: float = 1.0,
        sample_std: float = 0.05,
        consistency_weight: float = 0.0,
        memory_bank_size: int = 0,
        use_amp: bool = False,
        policy_weight: float = 1.0,
        diagnostics_every: int = 0,
        num_samples: int = 1,
        train_gallery: bool = True,
        normalize_advantage: bool = False,
        memory_store: str = 'sample',
        policy: str = 'gaussian',
        pl_tau: float = 0.05,
        pl_topk: int = 100
    ):
        self.model = model
        self.reward_fn = reward_fn
        self.optimizer = optimizer
        self.logger = logger
        self.device = device
        self.reference_model = reference_model
        self.gradient_clip = gradient_clip
        self.sample_std = float(sample_std)
        self.consistency_weight = float(consistency_weight)
        self.memory_bank = EmbeddingMemoryBank(capacity=memory_bank_size) if memory_bank_size > 0 else None
        self.use_amp = use_amp
        # policy_weight=0 gives the consistency-only control; diagnostics_every=N
        # measures policy vs consistency gradient norms separately every N steps
        # (two extra backward passes, so keep N large in real runs).
        self.policy_weight = float(policy_weight)
        self.diagnostics_every = int(diagnostics_every)
        # num_samples > 1 switches to the multi-sample estimator; memory_store
        # 'mean' keeps deterministic means in the bank instead of noisy samples.
        self.num_samples = int(num_samples)
        self.train_gallery = bool(train_gallery)
        self.normalize_advantage = bool(normalize_advantage)
        self.memory_store = memory_store
        # policy 'plackett_luce': rankings are the actions (training/pl_policy.py);
        # num_samples is then the number of sampled rankings per query.
        # policy 'softmax_control': differentiable control on the same scores, batch and bank.
        # 'smoothap_control' / 'fastap_bank': AP-surrogate controls on the same batch and bank
        # (pl_tau is the Smooth-AP sigmoid temperature; FastAP uses 10 bins).
        if policy not in ('gaussian', 'plackett_luce', 'softmax_control', 'smoothap_control', 'fastap_bank'):
            raise ValueError(f"Unknown policy: {policy}")
        self.policy, self.pl_tau, self.pl_topk = policy, float(pl_tau), int(pl_topk)

        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")

        self.global_step = 0

    @staticmethod
    def _freeze_bn(model: nn.Module):
        """Set all BatchNorm layers to eval mode (frozen running stats)."""
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()

    def _sample_on_sphere(
        self,
        mean_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample embeddings and compute log-prob on the unit sphere.

        Instead of computing log-prob in pre-normalization space (which is
        biased after L2 projection), we use the cosine distance between the
        sampled point and the mean on the sphere as a proxy log-prob:
            log π ∝ cos_sim(sample, mean) / σ²
        This is proportional to the log-density of a von Mises–Fisher
        distribution centered at the mean, which is the correct distribution
        family for points on a unit sphere.

        The sampled action is detached inside the log-prob: the score-function
        estimator differentiates log π(a|θ) with the action a held fixed.
        Leaving it attached adds a reparameterisation term that shrank the
        gradient ~20x at d=512, σ=0.01 (see scripts/audit_estimator.py).
        """
        noise = self.sample_std * torch.randn_like(mean_embeddings)
        raw = mean_embeddings + noise
        sampled = F.normalize(raw, p=2, dim=1)

        cos_sim = (sampled.detach() * mean_embeddings).sum(dim=1)
        log_prob = cos_sim / (self.sample_std ** 2)
        return sampled, log_prob

    def _build_gallery(
        self,
        batch_gallery: torch.Tensor,
        batch_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate current batch gallery with the memory bank."""
        if self.memory_bank is not None and len(self.memory_bank) > 0:
            mem_embs, mem_labels = self.memory_bank.get()
            return (
                torch.cat([batch_gallery, mem_embs], dim=0),
                torch.cat([batch_labels, mem_labels], dim=0),
            )
        return batch_gallery, batch_labels

    def train_epoch(
        self,
        train_loader,
        epoch: int,
        log_interval: int = 10,
        scheduler=None
    ) -> Dict[str, float]:
        self.model.train()
        self._freeze_bn(self.model)

        if self.memory_bank is not None:
            self.memory_bank.reset()

        reward_meter = AverageMeter()
        loss_meter = AverageMeter()
        baseline_meter = AverageMeter()
        diag_meters = {}

        start_time = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} (RL Training)")

        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            query_images = images[0::2]
            gallery_images = images[1::2]
            query_labels = labels[0::2]
            gallery_labels = labels[1::2]

            if query_images.size(0) == 0 or gallery_images.size(0) == 0:
                continue

            loss, metrics = self.train_step(
                query_images, gallery_images,
                query_labels, gallery_labels
            )

            loss_meter.update(loss.item(), query_images.size(0))
            reward_meter.update(metrics['reward'], query_images.size(0))
            baseline_meter.update(metrics['baseline'], query_images.size(0))
            for k, v in metrics.items():
                if k.startswith('diag/'):
                    diag_meters.setdefault(k, AverageMeter()).update(v)

            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'reward': f"{reward_meter.avg:.4f}",
                'baseline': f"{baseline_meter.avg:.4f}"
            })

            if scheduler is not None:
                scheduler.step()

            if batch_idx % log_interval == 0:
                self.logger.log_scalar('train/rl_loss', loss.item(), self.global_step)
                self.logger.log_scalar('train/reward', metrics['reward'], self.global_step)
                self.logger.log_scalar('train/baseline', metrics['baseline'], self.global_step)
                self.logger.log_scalar('train/advantage', metrics['advantage'], self.global_step)
                for k, v in metrics.items():
                    if k.startswith('diag/'):
                        self.logger.log_scalar(f'train/{k[5:]}', v, self.global_step)

            self.global_step += 1

        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch} completed in {format_time(elapsed_time)}")
        print(f"  Average Reward: {reward_meter.avg:.4f}")
        print(f"  Average Loss: {loss_meter.avg:.4f}")
        diag = {k[5:]: m.avg for k, m in diag_meters.items()}
        if diag:
            print("  Diagnostics (epoch means): " + ", ".join(f"{k}={v:.4g}" for k, v in sorted(diag.items())))

        return {
            'reward': reward_meter.avg,
            'loss': loss_meter.avg,
            'baseline': baseline_meter.avg,
            'time': elapsed_time,
            **diag,
        }

    def train_step(
        self,
        query_images: torch.Tensor,
        gallery_images: torch.Tensor,
        query_labels: torch.Tensor,
        gallery_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Paper-aligned REINFORCE step.

        1. Forward pass → deterministic means (policy centers).
        2. Draw sample #1 → compute per-query AP reward.
        3. Draw sample #2 → compute per-query AP baseline.
        4. Per-query advantage = reward_i - baseline_i.
        5. Per-query log-prob (on sphere).
        6. Policy loss = -mean(advantage_i * log_prob_i).
        """
        self.optimizer.zero_grad()

        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            query_means = self.model(query_images)
            gallery_means = self.model(gallery_images)

        # Sampling, ranking and log-probs in fp32: σ is tiny and κ = 1/σ² is large.
        query_means = query_means.float()
        gallery_means = gallery_means.float()

        extra = {}
        if self.policy in ('plackett_luce', 'softmax_control', 'smoothap_control', 'fastap_bank'):
            # Every batch item ranks all others plus the memory bank of means.
            all_means = torch.cat([query_means, gallery_means])
            all_labels = torch.cat([query_labels, gallery_labels])
            mem_emb, mem_labels = (self.memory_bank.get() if self.memory_bank is not None
                                   and len(self.memory_bank) > 0 else (None, None))
            if self.policy == 'plackett_luce':
                policy_loss, ms = pl_policy_loss(all_means, all_labels, self.pl_tau, self.num_samples,
                                                 self.pl_topk, mem_emb, mem_labels)
            elif self.policy == 'softmax_control':
                policy_loss, ms = softmax_control_loss(all_means, all_labels, self.pl_tau, self.pl_topk,
                                                       mem_emb, mem_labels)
                ms['adv_q_abs'] = 0.0
            elif self.policy == 'smoothap_control':
                policy_loss, ms = smoothap_control_loss(all_means, all_labels, self.pl_tau, self.pl_topk,
                                                        mem_emb, mem_labels)
                ms['adv_q_abs'] = 0.0
            else:
                policy_loss, ms = fastap_bank_loss(all_means, all_labels, 10, self.pl_topk, mem_emb, mem_labels)
                ms['adv_q_abs'] = 0.0
            per_query_reward = torch.full((query_means.size(0),), ms['reward'], device=self.device)
            per_query_baseline = per_query_reward
            per_query_advantage = torch.full_like(per_query_reward, ms['adv_q_abs'])
            extra = {f'diag/{k}': v for k, v in ms.items() if k != 'reward'}
        elif self.num_samples > 1:
            # Multi-sample product rollouts with leave-one-out baselines and
            # gallery gradients (training/rl_estimators.py, PROGRESS D1).
            mem_emb, mem_labels = (self.memory_bank.get() if self.memory_bank is not None
                                   and len(self.memory_bank) > 0 else (None, None))
            policy_loss, ms = multi_sample_policy_loss(
                query_means, gallery_means, query_labels, gallery_labels,
                sigma=self.sample_std, num_samples=self.num_samples,
                mem_emb=mem_emb, mem_labels=mem_labels,
                sample_gallery=self.train_gallery, normalize_advantage=self.normalize_advantage)
            per_query_reward = torch.full((query_means.size(0),), ms['reward'], device=self.device)
            per_query_baseline = per_query_reward
            per_query_advantage = torch.full_like(per_query_reward, ms['adv_q_abs'])
            s_gallery = gallery_means.detach()   # memory stores fresh means
            extra = {f'diag/{k}': v for k, v in ms.items() if k != 'reward'}
        else:
            # --- Sample #1: reward sample ---
            s_query, q_log_prob = self._sample_on_sphere(query_means)
            s_gallery, _ = self._sample_on_sphere(gallery_means)

            full_gallery, full_gallery_labels = self._build_gallery(s_gallery, gallery_labels)

            per_query_reward = self.reward_fn.compute_per_example_reward(
                s_query, full_gallery, query_labels, full_gallery_labels
            )

            # --- Baseline (no grad): configured type is honoured in AMP and fp32 alike ---
            with torch.no_grad():
                if self.reward_fn.baseline_type == 'moving_average':
                    per_query_baseline = self.reward_fn.get_baseline().expand_as(per_query_reward)
                elif self.reward_fn.baseline_type == 'none':
                    per_query_baseline = torch.zeros_like(per_query_reward)
                elif self.reward_fn.baseline_type == 'sample':
                    # Independent second sample (paper's Algorithm 2 style)
                    b_query, _ = self._sample_on_sphere(query_means)
                    b_gallery, _ = self._sample_on_sphere(gallery_means)
                    b_full_gallery, b_full_labels = self._build_gallery(b_gallery, gallery_labels)
                    per_query_baseline = self.reward_fn.compute_per_example_reward(
                        b_query, b_full_gallery, query_labels, b_full_labels
                    )
                else:
                    raise ValueError(f"Unknown baseline type: {self.reward_fn.baseline_type}")

            per_query_advantage = per_query_reward - per_query_baseline
            if self.reward_fn.baseline_type == 'moving_average':
                self.reward_fn.update_baseline(per_query_reward.mean())

            # Policy loss: per-example REINFORCE
            policy_loss = -(per_query_advantage.detach() * q_log_prob).mean()

        # Consistency regulariser (anchors to Stage 1 model)
        consistency_loss = torch.tensor(0.0, device=self.device)
        if self.reference_model is not None and self.consistency_weight > 0:
            with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                ref_query = self.reference_model(query_images)
                ref_gallery = self.reference_model(gallery_images)
            consistency_loss = 0.5 * (
                F.mse_loss(query_means, ref_query.float()) +
                F.mse_loss(gallery_means, ref_gallery.float())
            )

        weighted_policy = self.policy_weight * policy_loss
        weighted_consistency = self.consistency_weight * consistency_loss
        total_loss = weighted_policy + weighted_consistency

        diag = {}
        if self.diagnostics_every and self.global_step % self.diagnostics_every == 0:
            params = [p for p in self.model.parameters() if p.requires_grad]
            for name, term in (('policy', weighted_policy), ('consistency', weighted_consistency)):
                if term.requires_grad:
                    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
                    diag[f'diag/grad_norm_{name}'] = torch.norm(torch.stack(
                        [g.norm() for g in grads if g is not None])).item()
                else:
                    diag[f'diag/grad_norm_{name}'] = 0.0

        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            total_loss.backward()
        # Pre-clip norm is always measured; clipping only if configured
        max_norm = self.gradient_clip if self.gradient_clip > 0 else float('inf')
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm).item()
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.memory_bank is not None:
            if self.policy in ('plackett_luce', 'softmax_control', 'smoothap_control', 'fastap_bank'):
                self.memory_bank.update(all_means.detach(), all_labels.detach())
            else:
                stored = gallery_means if self.memory_store == 'mean' else s_gallery
                self.memory_bank.update(stored.detach(), gallery_labels.detach())

        reward_mean = per_query_reward.mean().item()
        baseline_mean = per_query_baseline.mean().item()
        metrics = {
            'reward': reward_mean,
            'baseline': baseline_mean,
            'advantage': (reward_mean - baseline_mean),
            'diag/grad_norm_total': grad_norm,
            'diag/clipped': float(self.gradient_clip > 0 and grad_norm > self.gradient_clip),
            'diag/policy_loss': policy_loss.item(),
            'diag/consistency_loss': consistency_loss.item(),
            'diag/adv_abs_mean': per_query_advantage.abs().mean().item(),
            'diag/adv_nonzero_frac': (per_query_advantage.abs() > 1e-6).float().mean().item(),
            **diag,
            **extra,
        }
        return total_loss, metrics

    def validate(
        self,
        val_loader,
        epoch: int
    ) -> Dict[str, float]:
        """Deterministic evaluation — no sampling, no memory bank."""
        self.model.eval()

        reward_meter = AverageMeter()

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} (Validation)")

            for images, labels, _ in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                query_images = images[0::2]
                gallery_images = images[1::2]
                q_labels = labels[0::2]
                g_labels = labels[1::2]

                if query_images.size(0) == 0 or gallery_images.size(0) == 0:
                    continue

                query_emb = self.model(query_images)
                gallery_emb = self.model(gallery_images)

                reward = self.reward_fn.compute_reward(
                    query_emb, gallery_emb, q_labels, g_labels
                )

                reward_meter.update(reward.item(), query_images.size(0))
                pbar.set_postfix({'reward': f"{reward_meter.avg:.4f}"})

        print(f"Validation Reward: {reward_meter.avg:.4f}")
        self.logger.log_scalar('val/reward', reward_meter.avg, epoch)

        return {'reward': reward_meter.avg}

