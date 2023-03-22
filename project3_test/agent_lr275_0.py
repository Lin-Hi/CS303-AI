import copy
import math

import numpy
import torch
import torch.nn as nn
from typing import Tuple
from time import time
import torch.nn.functional as F
from os import path

P = 3  # spline degree
N_CTPS = 5  # number of control points

RADIUS = 0.3
N_CLASSES = 10
FEATURE_DIM = 256


class Agent:
    def __init__(self) -> None:
        """Initialize the agent, e.g., load the classifier model. """
        # TODO: prepare your agent here
        self.net = Net()
        PATH = path.join(path.dirname(__file__), 'classifier_para.pth')
        self.net.load_state_dict(torch.load(PATH))

    def get_action(self,
                   target_pos: torch.Tensor,
                   target_features: torch.Tensor,
                   class_scores: torch.Tensor,
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the parameters required to fire a projectile.

        Args:
            target_pos: x-y positions of shape `(N, 2)` where `N` is the number of targets.
            target_features: features of shape `(N, d)`.
            class_scores: scores associated with each class of targets. `(K,)` where `K` is the number of classes.
        Return: Tensor of shape `(N_CTPS-2, 2)`
            the second to the second last control points
        """
        start_time = time()

        assert len(target_pos) == len(target_features)

        outputs = self.net(target_features)
        _, target_cls = torch.max(outputs, 1)
        # TODO: compute the firing speed and angle that would give the best score.
        # Example: return a random configuration

        # best_score = float('-inf')
        # best_ans = None
        # while time() - start_time < 0.25:
        #     random_ans = torch.rand((3, 2)) * torch.tensor([3, 2.]) + torch.tensor([1., -1.])
        #     score_real = evaluate(compute_traj(random_ans), target_pos, class_scores[target_cls], RADIUS)
        #     if score_real.data > best_score:
        #         best_score = score_real.data
        #         best_ans = random_ans
        # print(best_score.data)
        # return best_ans
        best_ans = None
        best_score = -math.inf
        while time() - start_time < 0.20:
            random_ans = torch.rand((N_CTPS-2, 2)) * torch.tensor([5, 7]) + torch.tensor([0, -3.5])
            score_real = evaluate(compute_traj(random_ans), target_pos, class_scores[target_cls], RADIUS)
            if score_real.item() > best_score:
                best_score = score_real.item()
                best_ans = copy.deepcopy(random_ans)

        step_length = 0.9
        if best_score <= 0:
            step_length = 2.75
        ctps_inter = copy.deepcopy(best_ans)
        ctps_inter.requires_grad = True
        while time() - start_time < 0.29:
            score_modified = evaluate_modified(compute_traj(ctps_inter), target_pos, class_scores[target_cls], RADIUS)
            score_modified.backward()
            ctps_inter.data = ctps_inter.data + step_length * ctps_inter.grad / torch.norm(ctps_inter.grad)
            score_real = evaluate(compute_traj(ctps_inter), target_pos, class_scores[target_cls], RADIUS)
            step_length = step_length * 0.95
            if score_real.item() > best_score:
                best_score = score_real.item()
                best_ans = copy.deepcopy(ctps_inter)
        return best_ans


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)  # 线性变换
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def compute_traj(ctps_inter: torch.Tensor):
    """Compute the discretized trajectory given the second to the second control points"""
    t = torch.linspace(0, N_CTPS - P, 100, device=ctps_inter.device)
    knots = torch.cat([
        torch.zeros(P, device=ctps_inter.device),
        torch.arange(N_CTPS + 1 - P, device=ctps_inter.device),
        torch.full((P,), N_CTPS - P, device=ctps_inter.device),
    ])
    ctps = torch.cat([
        torch.tensor([[0., 0.]], device=ctps_inter.device),
        ctps_inter,
        torch.tensor([[N_CTPS, 0.]], device=ctps_inter.device)
    ])
    return splev(t, knots, ctps, P)


def evaluate(
        traj: torch.Tensor,
        target_pos: torch.Tensor,
        target_scores: torch.Tensor,
        radius: float,
) -> torch.Tensor:
    """Evaluate the trajectory and return the score it gets.

    Parameters
    ----------
    traj: Tensor of shape `(*, T, 2)`
        The discretized trajectory, where `*` is some batch dimension and `T` is the discretized time dimension.
    target_pos: Tensor of shape `(N, 2)`
        x-y positions of shape where `N` is the number of targets.
    target_scores: Tensor of shape `(N,)`
        Scores you get when the corresponding targets get hit.
    """
    cdist = torch.cdist(target_pos, traj)  # see https://pytorch.org/docs/stable/generated/torch.cdist.html
    d = cdist.min(-1).values
    hit = (d < radius)
    value = torch.sum(hit * target_scores, dim=-1)
    return value


def splev(
        x: torch.Tensor,
        knots: torch.Tensor,
        ctps: torch.Tensor,
        degree: int,
        der: int = 0
) -> torch.Tensor:
    """Evaluate a B-spline or its derivatives.

    See https://en.wikipedia.org/wiki/B-spline for more about B-Splines.
    This is a PyTorch implementation of https://en.wikipedia.org/wiki/De_Boor%27s_algorithm

    Parameters
    ----------
    x : Tensor of shape `(t,)`
        An array of points at which to return the value of the smoothed
        spline or its derivatives.
    knots: Tensor of shape `(m,)`
        A B-Spline is a piece-wise polynomial.
        The values of x where the pieces of polynomial meet are known as knots.
    ctps: Tensor of shape `(n_ctps, dim)`
        Control points of the spline.
    degree: int
        Degree of the spline.
    der: int, optional
        The order of derivative of the spline to compute (must be less than
        or equal to k, the degree of the spline).
    """
    if der == 0:
        return _splev_torch_impl(x, knots, ctps, degree)
    else:
        assert der <= degree, "The order of derivative to compute must be less than or equal to k."
        n = ctps.size(-2)
        ctps = (ctps[..., 1:, :] - ctps[..., :-1, :]) / (knots[degree + 1:degree + n] - knots[1:n]).unsqueeze(-1)
        return degree * splev(x, knots[..., 1:-1], ctps, degree - 1, der - 1)


def _splev_torch_impl(x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, k: int):
    """
        x: (t,)
        t: (m, )
        c: (n_ctps, dim)
    """
    assert t.size(0) == c.size(0) + k + 1, f"{len(t)} != {len(c)} + {k} + {1}"  # m= n + k + 1

    x = torch.atleast_1d(x)
    assert x.dim() == 1 and t.dim() == 1 and c.dim() == 2, f"{x.shape}, {t.shape}, {c.shape}"
    n = c.size(0)
    u = (torch.searchsorted(t, x) - 1).clip(k, n - 1).unsqueeze(-1)
    x = x.unsqueeze(-1)
    d = c[u - k + torch.arange(k + 1, device=c.device)].contiguous()
    for r in range(1, k + 1):
        j = torch.arange(r - 1, k, device=c.device) + 1
        t0 = t[j + u - k]
        t1 = t[j + u + 1 - r]
        alpha = ((x - t0) / (t1 - t0)).unsqueeze(-1)
        d[:, j] = (1 - alpha) * d[:, j - 1] + alpha * d[:, j]
    return d[:, k]


def evaluate_modified(
        traj: torch.Tensor,
        target_pos: torch.Tensor,
        target_scores: torch.Tensor,
        radius: float,
) -> torch.Tensor:
    """Evaluate the trajectory and return the score it gets.

    Parameters
    ----------
    traj: Tensor of shape `(*, T, 2)`
        The discretized trajectory, where `*` is some batch dimension and `T` is the discretized time dimension.
    target_pos: Tensor of shape `(N, 2)`
        x-y positions of shape where `N` is the number of targets.
    target_scores: Tensor of shape `(N,)`
        Scores you get when the corresponding targets get hit.
    """
    cdist = torch.cdist(target_pos, traj)  # see https://pytorch.org/docs/stable/generated/torch.cdist.html
    d = cdist.min(-1).values
    hit = (d < radius)
    d[hit] = 1
    d[~hit] = RADIUS / d[~hit]
    value = torch.sum(d * target_scores, dim=-1)
    return value
