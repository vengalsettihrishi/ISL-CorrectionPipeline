"""CTC decoding and edit-distance metrics."""

from __future__ import annotations

from typing import List, Sequence

import torch


def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor, blank_id: int = 0) -> List[List[int]]:
    pred = log_probs.argmax(dim=-1)
    decoded: List[List[int]] = []
    for b in range(pred.size(0)):
        raw = pred[b, : int(lengths[b])].tolist()
        collapsed = []
        prev = None
        for token in raw:
            if token != prev:
                collapsed.append(token)
            prev = token
        decoded.append([t for t in collapsed if t != blank_id])
    return decoded


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def error_rate(refs: Sequence[Sequence], hyps: Sequence[Sequence]) -> float:
    edits = 0
    total = 0
    for ref, hyp in zip(refs, hyps):
        edits += edit_distance(ref, hyp)
        total += max(len(ref), 1)
    return edits / max(total, 1)

