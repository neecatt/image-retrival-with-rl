"""
Graded-relevance fine-tuning on candidate-list batches (Phase 2 of docs/RL_RESEARCH_PLAN.md).

Two arms that see exactly the same batches and grades:
  mode='rl'          multi-sample REINFORCE (product rollouts, leave-one-out baselines,
                     query and gallery sampled) with exact nDCG@k as reward, plus the
                     consistency anchor to the initial model
  mode='approxndcg'  graded differentiable control: ApproxNDCG (Qin et al. 2010)
Unverified same-landmark items and the query itself are masked out of every ranking.
A memory bank of recent mean embeddings adds other-landmark negatives (grade 0);
bank entries from a query's own landmark are masked.
"""

import time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.graded_lists import batch_gains
from losses.graded import approx_ndcg_loss, ndcg_at_k
from training.rl_estimators import sample_sphere
from training.rl_trainer import EmbeddingMemoryBank


def graded_policy_loss(q_mean, g_mean, gains, valid, sigma, S, k,
                       mem_emb=None, mem_valid=None):
    """Product-rollout REINFORCE with exact nDCG@k reward. gains/valid: (B, N) over the batch gallery;
    mem_valid: (B, M) True for memory items usable as grade-0 negatives."""
    B, N = gains.shape
    q_s, q_logp = sample_sphere(q_mean, sigma, S)                       # (S,B,d), (S,B)
    g_s, g_logp = sample_sphere(g_mean, sigma, S)                       # (S,N,d), (S,N)
    with torch.no_grad():
        sim = torch.einsum('sbd,tnd->bstn', q_s.detach(), g_s.detach())  # (B,S,S,N)
        gn = gains[:, None, None, :].expand(B, S, S, N)
        neg_inf = torch.finfo(sim.dtype).min
        sim = sim.masked_fill(~valid[:, None, None, :], neg_inf)
        if mem_emb is not None and mem_emb.numel() > 0:
            sm = torch.einsum('sbd,md->bsm', q_s.detach(), mem_emb)
            sm = sm.masked_fill(~mem_valid[:, None, :], neg_inf)
            sim = torch.cat([sim, sm[:, :, None, :].expand(-1, -1, S, -1)], -1)
            gn = torch.cat([gn, torch.zeros(B, S, S, sm.size(-1), device=gn.device)], -1)
        R = ndcg_at_k(sim, gn, k)                                       # (B,S,S)
        r_q, r_g = R.mean(2), R.mean(1)
        A_q = r_q - (r_q.sum(1, keepdim=True) - r_q) / (S - 1)
        A_g = r_g - (r_g.sum(1, keepdim=True) - r_g) / (S - 1)
    loss = -(A_q * q_logp.t()).mean() - (A_g * g_logp.sum(1)[None, :]).mean()
    stats = {'reward': R.mean().item(), 'adv_q_abs': A_q.abs().mean().item(),
             'reward_std_within_query': R.flatten(1).std(1).mean().item()}
    return loss, stats


class GradedTrainer:
    def __init__(self, model: nn.Module, optimizer, device, grades: Dict, labels_all: torch.Tensor,
                 mode: str, queries_per_batch: int, candidates_per_query: int,
                 reference_model: Optional[nn.Module] = None, consistency_weight: float = 0.0,
                 sample_std: float = 0.05, num_samples: int = 16, ndcg_k: int = 10,
                 approx_tau: float = 0.05, memory_bank_size: int = 4096, use_amp: bool = False,
                 logger=None):
        assert mode in ('rl', 'approxndcg')
        self.model, self.opt, self.device, self.grades = model, optimizer, device, grades
        self.labels_all = labels_all
        self.mode, self.B, self.C = mode, queries_per_batch, candidates_per_query
        self.ref, self.cw = reference_model, float(consistency_weight)
        self.sigma, self.S, self.k, self.tau = float(sample_std), int(num_samples), int(ndcg_k), float(approx_tau)
        self.memory = EmbeddingMemoryBank(memory_bank_size) if memory_bank_size > 0 else None
        self.use_amp, self.logger, self.step = use_amp, logger, 0

    @staticmethod
    def _freeze_bn(model):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()

    def train_epoch(self, loader, epoch: int, scheduler=None, log_every: int = 50) -> Dict[str, float]:
        self.model.train()
        self._freeze_bn(self.model)
        if self.memory is not None:
            self.memory.reset()
        if hasattr(loader.batch_sampler, 'set_epoch'):
            loader.batch_sampler.set_epoch(epoch)
        sums, n, t0 = {}, 0, time.time()
        for images, labels, indices in loader:
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device)
            q_pos, gains, valid = batch_gains(indices, labels.cpu(), self.grades, self.B, self.C)
            q_pos, gains, valid = q_pos.to(self.device), gains.to(self.device), valid.to(self.device)

            self.opt.zero_grad()
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                emb = self.model(images)
            emb = emb.float()
            q_emb = emb[q_pos]
            mem_emb = mem_valid = None
            if self.memory is not None and len(self.memory) > 0:
                mem_emb, mem_lab = self.memory.get()
                mem_valid = mem_lab[None, :] != labels[q_pos][:, None]

            if self.mode == 'rl':
                loss, st = graded_policy_loss(q_emb, emb, gains, valid, self.sigma, self.S, self.k, mem_emb, mem_valid)
                if self.ref is not None and self.cw > 0:
                    with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                        ref = self.ref(images).float()
                    cons = F.mse_loss(emb, ref)
                    loss = loss + self.cw * cons
                    st['consistency_loss'] = cons.item()
            else:
                sim = q_emb @ emb.t()
                gn = gains
                if mem_emb is not None:
                    sim = torch.cat([sim, q_emb @ mem_emb.t()], 1)
                    gn = torch.cat([gains, torch.zeros_like(sim[:, gains.size(1):])], 1)
                    vmask = torch.cat([valid, mem_valid], 1)
                else:
                    vmask = valid
                sim = sim.masked_fill(~vmask, -1e4)
                loss = approx_ndcg_loss(sim, gn, self.tau)
                with torch.no_grad():
                    st = {'reward': ndcg_at_k(sim.detach(), gn, self.k).mean().item()}

            loss.backward()
            st['grad_norm'] = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf')).item()
            self.opt.step()
            if scheduler is not None:
                scheduler.step()
            if self.memory is not None:
                self.memory.update(emb.detach(), labels.detach())
            st['loss'] = loss.item()
            for kk, v in st.items():
                sums[kk] = sums.get(kk, 0.0) + v
            n += 1
            if self.logger is not None and self.step % log_every == 0:
                for kk, v in st.items():
                    self.logger.log_scalar(f'train/{kk}', v, self.step)
            if self.step % log_every == 0:
                print(f"  step {self.step}: " + ", ".join(f"{kk}={v:.4g}" for kk, v in st.items()), flush=True)
            self.step += 1
        out = {kk: v / max(n, 1) for kk, v in sums.items()}
        out['time'] = time.time() - t0
        print(f"Epoch {epoch} ({self.mode}) done in {out['time'] / 60:.1f} min: "
              + ", ".join(f"{kk}={v:.4g}" for kk, v in out.items() if kk != 'time'), flush=True)
        return out
