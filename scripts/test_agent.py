import argparse
import os
import json
import numpy as np
import torch
from src.env.mountain_car import CustomMountainCar
from src.agents.vin_agent import VINAgent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_path", type=str, default="../exp")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--num_eps", type=int, default=30)
    arglist = parser.parse_args()
    return arglist

def episode(env, agent, max_steps=500):
    agent.eval()
    agent.reset()

    data = {"obs": [], "act": [], "reward": [], "next_obs": [], "done": []}
    obs = env.reset()
    
    for t in range(max_steps):
        # s = env.obs2state(obs)[0]
        obs_tensor = torch.from_numpy(obs).view(1, -1).to(torch.float32)
        a = agent.choose_action(obs_tensor).numpy()[0]
        # a = torch.multinomial(pi, 1).numpy()[0]

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

def main(arglist):
    np.random.seed(arglist.seed)
    torch.manual_seed(arglist.seed)

    exp_path = os.path.join(arglist.exp_path, arglist.exp_name)

    # load args
    with open(os.path.join(exp_path, "args.json"), "r") as f:
        config = json.load(f)
    
    # load state dict
    state_dict = torch.load(os.path.join(exp_path, "model.pt"), map_location=torch.device("cpu"))
    state_dict = {k.replace("agent.", ""): v for (k, v) in state_dict.items() if "agent." in k}

    obs_dim = 2
    act_dim = 3
    agent = VINAgent(
        config["state_dim"], act_dim, obs_dim, config["hmm_rank"], config["horizon"],
        config["alpha"], config["epsilon"], config["obs_cov"]
    )
    agent.load_state_dict(state_dict, strict=True)
    print(agent)

    x_bins = 20
    v_bins = 20
    env = CustomMountainCar(x_bins=x_bins, v_bins=v_bins, seed=arglist.seed)
    
    scores = []
    for e in range(arglist.num_eps):
        data = episode(env, agent)
        scores.append(len(data['reward']))
        print(f"score: {scores[-1]}")

    print(f"scores mean: {np.mean(scores), np.std(scores)}")

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)