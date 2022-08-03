import argparse
import os
import pickle
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 

from src.env.mountain_car import CustomMountainCar
from src.agents.vin_agent import VINAgent
from src.algo.irl import DAC
from src.algo.firl import FIRL
from src.algo.rl_utils import train

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_path", type=str, default="../exp")
    parser.add_argument("--cp_path", type=str, default="none", help="checkpoint path, default=none")
    # agent args
    parser.add_argument("--state_dim", type=int, default=30)
    parser.add_argument("--hmm_rank", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--obs_cov", type=str, choices=["full", "diag", "tied"], default="full")
    # algo args
    parser.add_argument("--algo", type=str, choices=["dac", "firl"], default="dac")
    parser.add_argument("--hidden_dim", type=int, default=64, help="neural network hidden dims, default=64")
    parser.add_argument("--num_hidden", type=int, default=2, help="number of hidden layers, default=2")
    parser.add_argument("--activation", type=str, default="relu", help="neural network activation, default=relu")
    parser.add_argument("--gamma", type=float, default=0.9, help="trainer discount factor, default=0.9")
    parser.add_argument("--beta", type=float, default=0.1, help="softmax temperature, default=0.1")
    parser.add_argument("--polyak", type=float, default=0.995, help="polyak averaging factor, default=0.995")
    parser.add_argument("--norm_obs", type=bool_, default=False, help="whether to normalize observations for agent and algo, default=False")
    parser.add_argument("--use_state", type=bool_, default=False, help="whether to use state in discriminator and critic, default=False")
    # training args
    parser.add_argument("--buffer_size", type=int, default=1e4, help="agent replay buffer size, default=1e5")
    parser.add_argument("--d_batch_size", type=int, default=200, help="training batch size, default=200")
    parser.add_argument("--a_batch_size", type=int, default=32, help="actor critic batch size")
    parser.add_argument("--rnn_len", type=int, default=15, help="recurrent steps for training, default=15")
    parser.add_argument("--reward_steps", type=int, default=500, help="number of steps to compute reward loss in firl, default=500")
    parser.add_argument("--d_steps", type=int, default=30, help="discriminator steps, default=30")
    parser.add_argument("--a_steps", type=int, default=30, help="actor critic steps, default=30")
    parser.add_argument("--lr_d", type=float, default=0.001, help="discriminator learning rate, default=0.001")
    parser.add_argument("--lr_a", type=float, default=0.005, help="actor learning rate, default=0.001")
    parser.add_argument("--lr_c", type=float, default=0.001, help="critic learning rate, default=0.001")
    parser.add_argument("--decay", type=float, default=1e-5, help="weight decay, default=0")
    parser.add_argument("--grad_clip", type=float, default=1000., help="gradient clipping, default=1000.")
    parser.add_argument("--grad_penalty", type=float, default=0.1, help="discriminator gradient penalty, default=0.1")
    parser.add_argument("--bc_penalty", type=float, default=1., help="behavior cloning penalty, default=1.")
    parser.add_argument("--obs_penalty", type=float, default=0.1, help="observation penalty, default=0.1")
    # rollout args
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs, default=10")
    parser.add_argument("--max_steps", type=int, default=500, help="max steps per episode, default=500")
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--update_after", type=int, default=1000)
    parser.add_argument("--update_every", type=int, default=50)
    parser.add_argument("--cp_every", type=int, default=1000, help="checkpoint interval, default=1000")
    parser.add_argument("--verbose", type=bool_, default=True)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

def plot_history(df_history, plot_keys, plot_std=True):
    """ Plot learning history
    
    Args:
        df_history (pd.dataframe): learning history dataframe with a binary train column.
        plot_keys (list): list of colnames to be plotted.
        plot_std (bool): whether to plot std shade.

    Returns:
        fig (plt.figure)
        ax (plt.axes)
    """
    num_cols = len(plot_keys)
    width = min(4 * num_cols, 15)
    fig, ax = plt.subplots(1, num_cols, figsize=(width, 4))
    for i in range(num_cols):
        ax[i].plot(df_history["epoch"], df_history[plot_keys[i]], label="train")
        if plot_std:
            std = df_history[plot_keys[i].replace("_avg", "_std")]
            ax[i].fill_between(
                df_history["epoch"],
                df_history[plot_keys[i]] - std,
                df_history[plot_keys[i]] + std,
                alpha=0.4
            )

        ax[i].legend()
        ax[i].set_xlabel("epoch")
        ax[i].set_ylabel(plot_keys[i])
        ax[i].grid()
    
    plt.tight_layout()
    return fig, ax

class SaveCallback:
    def __init__(self, arglist, plot_keys, cp_history=None):
        date_time = datetime.datetime.now().strftime("%m-%d-%Y %H-%M-%S")
        exp_path = os.path.join(arglist.exp_path)
        save_path = os.path.join(exp_path, date_time)
        model_path = os.path.join(save_path, "models") # used to save model checkpoint
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        
        # save args
        with open(os.path.join(save_path, "args.json"), "w") as f:
            json.dump(vars(arglist), f)

        self.save_path = save_path
        self.model_path = model_path
        self.plot_keys = plot_keys
        self.cp_history = cp_history
        self.cp_every = arglist.cp_every
        self.iter = 0

    def __call__(self, model, logger):
        # save history
        df_history = pd.DataFrame(logger.history)
        if self.cp_history is not None:
            df_history["epoch"] += self.cp_history["epoch"].values[-1] + 1
            df_history["time"] += self.cp_history["time"].values[-1]
            df_history = pd.concat([self.cp_history, df_history], axis=0)
        df_history.to_csv(os.path.join(self.save_path, "history.csv"), index=False)

        # save history plot
        fig_history, _ = plot_history(df_history, self.plot_keys)
        fig_history.savefig(os.path.join(self.save_path, "history.png"), dpi=100)
        
        plt.clf()
        plt.close()

        if (self.iter + 1) % self.cp_every == 0:
            # save model
            torch.save(model.state_dict(), os.path.join(self.model_path, f"model_{self.iter}.pt"))
            print(f"\ncheckpoint saved at: {self.save_path}\n")
        self.iter += 1
    
    def save_model(self, model):
        torch.save(model.state_dict(), os.path.join(self.save_path, "model.pt"))

def main(arglist):
    np.random.seed(arglist.seed)
    torch.manual_seed(arglist.seed)
    
    # load expert data 
    data_path = "../data/data.p"
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)
    print(f"loaded {len(dataset)} episodes of demonstrations")
    
    env = CustomMountainCar()
    obs_dim = env.observation_space.low.shape[0]
    act_dim = env.action_space.n
    
    agent = VINAgent(
        arglist.state_dim, act_dim, obs_dim, arglist.hmm_rank, 
        arglist.horizon, obs_cov=arglist.obs_cov
    )
    
    if arglist.algo == "dac":
        model = DAC(
            agent, arglist.hidden_dim, arglist.num_hidden, arglist.activation,
            gamma=arglist.gamma, beta=arglist.beta, polyak=arglist.polyak,
            norm_obs=arglist.norm_obs, use_state=arglist.use_state, buffer_size=arglist.buffer_size,
            d_batch_size=arglist.d_batch_size, a_batch_size=arglist.a_batch_size, 
            rnn_len=arglist.rnn_len, d_steps=arglist.d_steps, a_steps=arglist.a_steps, 
            lr_d=arglist.lr_d, lr_a=arglist.lr_a, lr_c=arglist.lr_c, 
            decay=arglist.decay, grad_clip=arglist.grad_clip, 
            grad_penalty=arglist.grad_penalty, bc_penalty=arglist.bc_penalty, 
            obs_penalty=arglist.obs_penalty
        )
        plot_keys = ["eps_len_avg", "d_loss_avg", "critic_loss_avg", "actor_loss_avg", "bc_loss_avg", "obs_loss_avg"]
    elif arglist.algo == "firl":
        model = FIRL(
            agent, arglist.hidden_dim, arglist.num_hidden, arglist.activation,
            gamma=arglist.gamma, beta=arglist.beta, polyak=arglist.polyak,
            norm_obs=arglist.norm_obs, buffer_size=arglist.buffer_size,
            d_batch_size=arglist.d_batch_size, a_batch_size=arglist.a_batch_size, 
            rnn_len=arglist.rnn_len, reward_steps=arglist.reward_steps, 
            d_steps=arglist.d_steps, a_steps=arglist.a_steps, 
            lr_d=arglist.lr_d, lr_a=arglist.lr_a, lr_c=arglist.lr_c, 
            decay=arglist.decay, grad_clip=arglist.grad_clip, 
            grad_penalty=arglist.grad_penalty, obs_penalty=arglist.obs_penalty
        )
        plot_keys = ["eps_len_avg", "d_loss_avg", "r_loss_avg", "critic_loss_avg", "actor_loss_avg", "obs_loss_avg"]
    
    cp_history = None
    if arglist.cp_path != "none":
        cp_path = os.path.join(arglist.exp_path, arglist.cp_path)

        # load state dict
        state_dict = torch.load(os.path.join(cp_path, "model.pt"))

        # load history
        cp_history = pd.read_csv(os.path.join(cp_path, "history.csv"))

        model.load_state_dict(state_dict, strict=True)

        print(f"loaded checkpoint from {cp_path}\n")
    
    print(model)
    
    callback = None
    if arglist.save:
        callback = SaveCallback(arglist, plot_keys, cp_history=cp_history)
    
    model.fill_real_buffer(dataset)
    model, logger = train(
        env, model, arglist.epochs, max_steps=arglist.max_steps, 
        steps_per_epoch=arglist.steps_per_epoch, update_after=arglist.update_after, 
        update_every=arglist.update_every, verbose=arglist.verbose, callback=callback
    )
    callback.save_model(model)

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)