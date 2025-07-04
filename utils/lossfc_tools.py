import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple
import matplotlib.pyplot as plt
import os
import math
import cv2
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)


def calculate_correct(scores, labels):
    assert scores.size(0) == labels.size(0)
    _, pred = scores.max(dim=1)
    correct = torch.sum(pred.eq(labels)).item()
    return correct


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def step_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return 0.0


def get_current_consistency_weight(epoch, weight, rampup_length, rampup_type='step'):
    if rampup_type == 'step':
        rampup_func = step_rampup
    elif rampup_type == 'linear':
        rampup_func = linear_rampup
    elif rampup_type == 'sigmoid':
        rampup_func = sigmoid_rampup
    else:
        raise ValueError("Rampup schedule not implemented")

    return weight * rampup_func(epoch, rampup_length)



def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def decoupling_loss(f_a, f_b, f_c=None, lambda_neg=0.5):
    def norm(x):
        return (x - x.mean(0)) / (x.std(0) + 1e-6)

    # Positive sample alignment: f_a and f_b
    f_a_norm = norm(f_a)
    f_b_norm = norm(f_b)
    c_ab = torch.mm(f_a_norm.T, f_b_norm) / f_a_norm.size(0)

    on_diag = torch.diagonal(c_ab).add_(-1).pow_(2).mean()
    off_diag = off_diagonal(c_ab).pow_(2).mean()
    pos_loss = on_diag + 0.005 * off_diag

    # Negative sample suppression: f_c and f_a
    if f_c is not None:
        f_c_norm = norm(f_c)
        c_ca = torch.mm(f_c_norm.T, f_a_norm) / f_a_norm.size(0)
        neg_on_diag = torch.diagonal(c_ca).pow(2).mean()
        loss = pos_loss + lambda_neg * neg_on_diag
    else:
        loss = pos_loss

    return loss


def info_nce_loss(
        f_anchor: torch.Tensor,
        f_pos: torch.Tensor,
        f_neg: torch.Tensor,
        temperature: float = 0.1,
        hard_negative_weight: float = 1.0,
) -> torch.Tensor:
    f_anchor = F.normalize(f_anchor, p=2, dim=-1)
    f_pos = F.normalize(f_pos, p=2, dim=-1)
    f_neg = F.normalize(f_neg, p=2, dim=-1)

    sim_pos = F.cosine_similarity(f_anchor, f_pos, dim=-1) / temperature
    sim_neg = F.cosine_similarity(f_anchor, f_neg, dim=-1) / temperature

    sim_neg = hard_negative_weight * sim_neg
    stacked = torch.stack([sim_pos, sim_neg], dim=-1)
    log_denom = torch.logsumexp(stacked, dim=-1)
    log_prob = sim_pos - log_denom
    loss = -log_prob.mean()

    return loss


def contrastive_decoupling_loss(
        f_a: torch.Tensor,
        f_b: torch.Tensor,
        f_c: torch.Tensor,
        alpha: float = 1.0,
        temperature: float = 0.1,
        hard_negative_weight: float = 0.7
) -> torch.Tensor:
    # 1) Factorization (Barlow Twins) loss
    fac_loss = factorization_loss(f_a, f_b)

    # 2) InfoNCE loss (anchor=f_a, pos=f_b, neg=f_c)
    nce_loss = info_nce_loss(
        f_anchor=f_a,
        f_pos=f_b,
        f_neg=f_c,
        temperature=temperature,
        hard_negative_weight=hard_negative_weight
    )
    # Total loss = fac_loss + alpha * nce_loss
    total_loss = fac_loss + alpha * nce_loss

    return total_loss
