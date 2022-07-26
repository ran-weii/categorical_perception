import argparse
import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 

from src.env.mountain_car import CustomMountainCar
from src.algo.planning import value_iteration
from src.agents.vin_agent import VINAgent
from src.algo.rl import SAC
from src.algo.rl_utils import train

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    # agent args
    parser.add_argument("--state_dim", type=int, default=30)
    parser.add_argument("--hmm_rank", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--obs_cov", type=str, choices=["full", "diag"], default="full")
    # algo args
    parser.add_argument("--hidden_dim", type=int, default=64, help="neural network hidden dims, default=64")
    parser.add_argument("--num_hidden", type=int, default=2, help="number of hidden layers, default=2")
    parser.add_argument("--activation", type=str, default="relu", help="neural network activation, default=relu")
    parser.add_argument("--gamma", type=float, default=0.9, help="trainer discount factor, default=0.9")
    parser.add_argument("--beta", type=float, default=0.2, help="softmax temperature, default=0.2")
    parser.add_argument("--polyak", type=float, default=0.995, help="polyak averaging factor, default=0.995")
    parser.add_argument("--norm_obs", type=bool_, default=False, help="whether to normalize observations for agent and algo, default=False")
    # training args
    parser.add_argument("--batch_size", type=int, default=100, help="training batch size, default=100")
    parser.add_argument("--buffer_size", type=int, default=1e5, help="agent replay buffer size, default=1e5")
    parser.add_argument("--a_steps", type=int, default=10, help="actor critic steps, default=50")
    parser.add_argument("--lr_a", type=float, default=0.005, help="actor learning rate, default=0.001")
    parser.add_argument("--lr_c", type=float, default=0.001, help="critic learning rate, default=0.001")
    parser.add_argument("--decay", type=float, default=1e-5, help="weight decay, default=0")
    parser.add_argument("--grad_clip", type=float, default=1000., help="gradient clipping, default=1000.")
    parser.add_argument("--obs_penalty", type=float, default=0.1)
    # rollout args
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs, default=10")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--update_after", type=int, default=1000)
    parser.add_argument("--update_every", type=int, default=50)
    parser.add_argument("--verbose", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

class CustomRewardMountaincar:
    def __init__(self, x_bins=20, v_bins=20):
        self.env = CustomMountainCar(x_bins=x_bins, v_bins=v_bins)
        self.env.make_initial_distribution()
        self.env.make_transition_matrix()
        transition_matrix = torch.from_numpy(self.env.transition_matrix)
        reward = torch.from_numpy(self.env.reward)

        gamma = 0.99 # discount factor
        max_iter = 2000

        q, info = value_iteration(
            transition_matrix, reward, gamma, softmax=False, max_iter=max_iter
        )
        self.v = q.max(-1)[0]

    def __call__(self, obs):
        s = self.env.obs2state(obs)[0]
        r = self.v[s]
        return r

def main(arglist):
    np.random.seed(arglist.seed)
    torch.manual_seed(arglist.seed)
    
    env = CustomMountainCar()
    custom_reward = CustomRewardMountaincar()
    obs_dim = env.observation_space.low.shape[0]
    act_dim = env.action_space.n
    
    agent = VINAgent(
        arglist.state_dim, act_dim, obs_dim, arglist.hmm_rank, 
        arglist.horizon, obs_cov=arglist.obs_cov
    )
    
    model = SAC(
        agent, arglist.hidden_dim, arglist.num_hidden, arglist.activation,
        gamma=arglist.gamma, beta=arglist.beta, polyak=arglist.polyak,
        norm_obs=arglist.norm_obs, buffer_size=arglist.buffer_size,
        batch_size=arglist.batch_size, a_steps=arglist.a_steps, 
        lr_a=arglist.lr_a, lr_c=arglist.lr_c, 
        decay=arglist.decay, grad_clip=arglist.grad_clip
    )
    print(model)

    model, logger = train(
        env, model, arglist.epochs, max_steps=arglist.max_steps, 
        steps_per_epoch=arglist.steps_per_epoch, update_after=arglist.update_after, 
        update_every=arglist.update_every, custom_reward=custom_reward, verbose=arglist.verbose
    )

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)