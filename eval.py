from src import generate_game, N_CTPS, evaluate, compute_traj, RADIUS
# from agent import Agent
from project3 import Agent
from tqdm import tqdm # a convenient progress bar
import torch

N_EVALS = 2000

if __name__ == "__main__":
    n_targets = 40
    agent = Agent()

    # This is a example of what the evaluation procedure looks like.
    # The whole dataset is divided into a training set and a test set.
    # The training set (including `data` and `label`) is distributed to you.
    # But in the final evaluation we will use the test set.

    data = torch.load("data.pth")
    label = data["label"]
    feature = data["feature"]

    print(label)
    print(label.shape)
    print(feature)
    print(feature.shape)

    scores = []
    for game in tqdm(range(N_EVALS)):

        # the class information is unavailable at test time.
        target_pos, target_features, target_cls, class_scores = generate_game(n_targets, N_CTPS, feature, label)
        ctps_inter = agent.get_action(target_pos, target_features, class_scores)
        score = evaluate(compute_traj(ctps_inter), target_pos, class_scores[target_cls], RADIUS)
        scores.append(score)
        print(score)

    print(torch.stack(scores).float().mean())
    