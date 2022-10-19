import argparse
import os
import pickle
import numpy as np
import torch
from src.env.mountain_car import CustomMountainCar
from src.algo.planning import value_iteration

def episode(env, policy, max_steps=500):
    data = {"obs": [], "act": [], "reward": [], "done": []}
    obs = env.reset()
    done = False
    for t in range(max_steps):
        s = env.obs2state(obs)[0]
        a = torch.multinomial(policy[s], 1).numpy()[0]

        next_obs, reward, next_done, info = env.step(a)

        data["obs"].append(obs)
        data["act"].append(a)
        data["reward"].append(reward)
        data["done"].append(done)
        
        if next_done:
            break
        obs = next_obs
        done = next_done
    
    # collect final time step
    s = env.obs2state(next_obs)[0]
    a = torch.multinomial(policy[s], 1).numpy()[0]
    data["obs"].append(next_obs)
    data["act"].append(a)
    data["reward"].append(reward)
    data["done"].append(next_done)
    return data

def main(arglist):
    seed = 0
    x_bins = 20
    v_bins = 20
    env = CustomMountainCar(x_bins=x_bins, v_bins=v_bins, seed=seed)
    env.make_initial_distribution()
    env.make_transition_matrix()

    transition_matrix = torch.from_numpy(env.transition_matrix)
    reward = torch.from_numpy(env.reward)

    gamma = 0.99 # discount factor
    beta = arglist.beta # softmax temperature
    max_iter = 2000

    q_soft, info = value_iteration(
        transition_matrix, reward, gamma, softmax=True, alpha=beta, max_iter=max_iter
    )
    print(f"soft value iteration info: {info}")
    
    # expert policy
    policy = torch.softmax(beta * q_soft, dim=-1)

    num_eps = arglist.num_eps
    max_steps = 500
    dataset = []
    for i in range(num_eps):
        data = episode(env, policy, max_steps=max_steps)
        dataset.append(data)
    
    # print demonstration stats
    eps_len = [len(d["reward"]) for d in dataset]
    print(f"demonstration performance: min={np.min(eps_len)}, max={np.max(eps_len)}, mean={np.mean(eps_len):.2f}")
    
    save_path = "../data"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with open(os.path.join(save_path, "data.p"), "wb") as f:
        pickle.dump(dataset, f)

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=1000.)
    parser.add_argument("--num_eps", type=float, default=30)
    arglist = parser.parse_args()
    return arglist

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)