import torch
import torch.nn as nn
import time
import copy
from typing import Tuple
import os.path as osp

P = 3  # spline degree
N_CTPS = 5  # number of control points

RADIUS = 0.3
N_CLASSES = 10
FEATURE_DIM = 256


def generate_game(
        n_targets: int,
        n_ctps: int,
        feature: torch.Tensor,
        label: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """
    Randomly generate a task configuration.
    """
    assert len(feature) == len(label)

    sample_indices = torch.randperm(len(feature))[:n_targets]
    target_pos = torch.rand((n_targets, 2)) * torch.tensor([n_ctps - 2, 2.]) + torch.tensor([1., -1.])
    target_features = feature[sample_indices]
    target_cls = label[sample_indices]
    class_scores = torch.randint(-N_CLASSES, N_CLASSES, (N_CLASSES,))

    return target_pos, target_features, target_cls, class_scores


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

def evaluate_gradient(
        traj: torch.Tensor,
        target_pos: torch.Tensor,
        target_scores: torch.Tensor,
) -> torch.Tensor:
    cdist = torch.cdist(target_pos, traj)
    return torch.sum(1 / (1 + torch.e ** (-cdist.min(-1).values)) * target_scores, dim=-1)


class Classifier(nn.Module):
    def __init__(self, in_size=256, hidden_unite_1=64, hidden_unite_2=16):
        super(Classifier, self).__init__()
        self.Classify_learner1 = nn.Linear(in_size, hidden_unite_1)
        self.Classify_norm1 = nn.BatchNorm1d(hidden_unite_1)
        self.Classify_relu = nn.functional.relu
        self.Classify_learner2 = nn.Linear(hidden_unite_1, hidden_unite_2)
        self.Classify_norm1 = nn.BatchNorm1d(hidden_unite_2)

    def forward(self, v):
        layer1 = self.Classify_learner1(v.view(-1, 256))
        relu1 = self.Classify_relu(layer1)
        layer2 = self.Classify_learner2(relu1)
        relu2 = self.Classify_relu(layer2)
        return relu2

    def predict(self, x):
        return torch.max(self(x).data, dim=1)[1]


class Agent:

    def __init__(self) -> None:
        # TODO: prepare your agent here
        classify_data = torch.load(osp.join(osp.dirname(__file__), "classifier.pth"))
        self.gnn = Classifier()
        self.gnn.load_state_dict(classify_data)

    def one_random(self):
        res = torch.zeros(N_CTPS - 2, 2)
        res[0:2, 0].uniform_(0, 5)
        res[0:, 1].uniform_(-3.0, 3.0)
        res.requires_grad = True
        return res

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
        start_time = time.time()
        assert len(target_pos) == len(target_features)
        predict = self.gnn.predict(target_features)
        # TODO: compute the firing speed and angle that would give the best score.

        ctps_inter = self.one_random()
        best_score, best_inter = -99999.0, None
        cnt = 0
        while time.time() - start_time < 0.28:
            cnt += 1
            if cnt % 15 == 14:
                ctps_inter = self.one_random()

            real_score = evaluate(compute_traj(ctps_inter), target_pos, class_scores[predict], RADIUS).item()

            if real_score > best_score:
                best_score = real_score
                best_inter = copy.deepcopy(ctps_inter)
            elif real_score <= 0 and cnt % 15 < 12:
                ctps_inter = self.one_random()
                continue
            gra_score = evaluate_gradient(compute_traj(ctps_inter), target_pos, class_scores[predict])
            gra_score.backward()
            ctps_inter.data = ctps_inter.data - ctps_inter.grad / torch.norm(ctps_inter.grad)
        return best_inter
