import os
import pickle
import torch
from src.env.mountain_car import CustomMountainCar
from src.algo.planning import value_iteration

def episode(env, policy, max_steps=500):
    data = {"obs": [], "act": [], "reward": [], "next_obs": [], "done": []}
    obs = env.reset()
    for t in range(max_steps):
        s = env.obs2state(obs)[0]
        a = torch.multinomial(policy[s], 1).numpy()[0]

        next_obs, reward, done, into = env.step(a)

        data["obs"].append(obs)
        data["act"].append(a)
        data["reward"].append(reward)
        data["next_obs"].append(next_obs)
        data["done"].append(1 if done else 0)
        
        if done:
            break
        obs = next_obs
    return data

def main():
    seed = 0
    x_bins = 20
    v_bins = 20
    env = CustomMountainCar(x_bins=x_bins, v_bins=v_bins, seed=seed)
    env.make_initial_distribution()
    env.make_transition_matrix()

    transition_matrix = torch.from_numpy(env.transition_matrix)
    reward = torch.from_numpy(env.reward)

    gamma = 0.99 # discount factor
    alpha = 10 # softmax temperature
    max_iter = 2000

    q_soft, info = value_iteration(
        transition_matrix, reward, gamma, softmax=True, alpha=alpha, max_iter=max_iter
    )
    print(f"soft value iteration info: {info}")
    
    # expert policy
    beta = 100
    policy = torch.softmax(beta * q_soft, dim=-1)

    num_eps = 30
    max_steps = 500
    dataset = []
    for i in range(num_eps):
        data = episode(env, policy, max_steps=max_steps)
        dataset.append(data)

    save_path = "../data"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with open(os.path.join(save_path, "data.p"), "wb") as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    main()