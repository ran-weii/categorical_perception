import argparse
import os
import pickle
import json
import datetime
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch 
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from src.agents.vin_agent import VINAgent
from src.algo.bc import BehaviorCloning

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_path", type=str, default="../exp")
    # agent args
    parser.add_argument("--state_dim", type=int, default=30)
    parser.add_argument("--hmm_rank", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=1.)
    parser.add_argument("--epsilon", type=float, default=1.)
    parser.add_argument("--obs_cov", type=str, choices=["full", "diag", "tied"], default="full")
    # algo args
    parser.add_argument("--bptt_steps", type=int, default=30)
    parser.add_argument("--obs_penalty", type=float, default=0.)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decay", type=float, default=0.)
    parser.add_argument("--grad_clip", type=float, default=None)
    # train args
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    return arglist

class CustomDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        obs = torch.from_numpy(np.stack(data["obs"])).to(torch.float32)
        act = torch.from_numpy(np.stack(data["act"])).to(torch.float32).view(-1, 1)
        return {"obs": obs, "act": act}

def collate_fn(batch):
    """ Collate batch of dict to have the same sequence length """
    assert isinstance(batch[0], dict)
    keys = list(batch[0].keys())
    pad_batch = {k: pad_sequence([b[k] for b in batch]) for k in keys}
    mask = 1 - torch.all(pad_batch[keys[0]] == 0, dim=-1).to(torch.float32)
    return pad_batch, mask

def train_test_split(dataset, train_ratio, batch_size):
    num_train = np.ceil(len(dataset) * train_ratio).astype(int)
    shuffle_idx = np.arange(len(dataset))
    np.random.shuffle(shuffle_idx)
    
    train_set = [dataset[i] for i in shuffle_idx[:num_train]]
    test_set = [dataset[i] for i in shuffle_idx[num_train:]]
    
    train_loader = DataLoader(
        CustomDataset(train_set), batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        CustomDataset(test_set), batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, test_loader

def train(model, train_loader, test_loader, epochs, verbose=1):
    history = []
    start_time = time.time()
    for e in range(epochs):
        train_stats = model.run_epoch(train_loader, train=True)
        test_stats = model.run_epoch(test_loader, train=False)
        
        tnow = time.time() - start_time
        train_stats.update({"epoch": e, "time": tnow})
        test_stats.update({"epoch": e, "time": tnow})
        history.append(train_stats)
        history.append(test_stats)

        if (e + 1) % verbose == 0:
            s = model.stdout(train_stats, test_stats)
            print("e: {}/{}, {}, t: {:.2f}".format(e + 1, epochs, s, tnow))

    df_history = pd.DataFrame(history)
    return model, df_history

def plot_history(df_history, plot_keys):
    """ Plot learning history
    
    Args:
        df_history (pd.dataframe): learning history dataframe with a binary train column.
        plot_keys (list): list of colnames to be plotted.
        plot_std (bool): whether to plot std shade.

    Returns:
        fig (plt.figure)
        ax (plt.axes)
    """
    df_train = df_history.loc[df_history["train"] == 1]
    df_test = df_history.loc[df_history["train"] == 0]

    num_cols = len(plot_keys)
    width = min(4 * num_cols, 15)
    fig, ax = plt.subplots(1, num_cols, figsize=(width, 4))
    for i in range(num_cols):
        ax[i].plot(df_train["epoch"], df_train[plot_keys[i]], label="train")
        ax[i].plot(df_test["epoch"], df_test[plot_keys[i]], label="test")

        ax[i].legend()
        ax[i].set_xlabel("epoch")
        ax[i].set_ylabel(plot_keys[i])
        ax[i].grid()
    
    plt.tight_layout()
    return fig, ax

def main(arglist):
    np.random.seed(arglist.seed)
    torch.manual_seed(arglist.seed)
    
    # load expert data 
    data_path = "../data/data.p"
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)
    print(f"loaded {len(dataset)} episodes of demonstrations")

    train_loader, test_loader = train_test_split(
        dataset, arglist.train_ratio, arglist.batch_size
    )
    print(f"train size: {len(train_loader.dataset)}, test size: {len(test_loader.dataset)}")
    
    # compute obs mean and variance
    obs_cat = np.vstack([d["obs"] for d in dataset])
    obs_mean = obs_cat.mean(axis=0)
    obs_variance = obs_cat.var(axis=0)
    
    obs_dim = 2
    act_dim = 3
    agent = VINAgent(
        arglist.state_dim, act_dim, obs_dim, arglist.hmm_rank, arglist.horizon, 
        arglist.alpha, arglist.epsilon, arglist.obs_cov
    )
    agent.obs_model.bn.moving_mean.data = torch.from_numpy(obs_mean).to(torch.float32)
    agent.obs_model.bn.moving_variance.data = torch.from_numpy(obs_variance).to(torch.float32)
    
    model = BehaviorCloning(
        agent, arglist.bptt_steps, arglist.obs_penalty, 
        arglist.lr, arglist.decay, arglist.grad_clip
    )
    print(model)

    model, df_history = train(model, train_loader, test_loader, arglist.epochs)

    # save results
    if arglist.save:
        date_time = datetime.datetime.now().strftime("%m-%d-%Y %H-%M-%S")
        exp_path = os.path.join(arglist.exp_path)
        save_path = os.path.join(exp_path, date_time)
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        # save args
        with open(os.path.join(save_path, "args.json"), "w") as f:
            json.dump(vars(arglist), f)
        
        # save model
        torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
        
        # save history
        df_history.to_csv(os.path.join(save_path, "history.csv"), index=False)
        
        # save history plot
        fig_history, _ = plot_history(df_history, model.loss_keys)
        fig_history.savefig(os.path.join(save_path, "history.png"), dpi=100)
        
        print(f"\nmodel saved at: {save_path}")

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)