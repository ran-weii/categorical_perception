import math
import torch
import torch.nn as nn
import torch.jit as jit

from typing import Union, Tuple
from torch import Tensor

class QMDPLayer(nn.Module):
    def __init__(self, state_dim, act_dim, rank, horizon):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.rank = rank
        self.horizon = horizon
        self.eps = 1e-6

        self.b0 = nn.Parameter(torch.randn(1, state_dim))
        self.tau = nn.Parameter(torch.randn(1, 1))
        
        if rank != 0:
            self.u = nn.Parameter(torch.randn(1, rank, state_dim)) # source tensor
            self.v = nn.Parameter(torch.randn(1, rank, state_dim)) # sink tensor
            self.w = nn.Parameter(torch.randn(1, rank, act_dim)) # action tensor 
        else:
            self.u = nn.Parameter(torch.randn(1, 1)) # dummy tensor
            self.v = nn.Parameter(torch.randn(1, 1)) # dummy tensor
            self.w = nn.Parameter(torch.randn(1, act_dim, state_dim, state_dim)) # transition tensor 

        nn.init.xavier_normal_(self.b0, gain=1.)
        nn.init.uniform_(self.tau, a=-1, b=1)
        nn.init.xavier_normal_(self.u, gain=1.)
        nn.init.xavier_normal_(self.v, gain=1.)
        nn.init.xavier_normal_(self.w, gain=1.)
    
    def __repr__(self):
        s = "{}(state_dim={}, act_dim={}, rank={}, horizon={})".format(
            self.__class__.__name__, self.state_dim, self.act_dim, self.rank, self.horizon
        )
        return s
    
    def compute_transition(self):
        """ Return transition matrix. size=[1, act_dim, state_dim, state_dim] """
        if self.rank != 0:
            w = torch.einsum("nri, nrj, nrk -> nkij", self.u, self.v, self.w)
        else:
            w = self.w
        return torch.softmax(w, dim=-1)
    
    def compute_value(self, transition: Tensor, reward: Tensor) -> Tensor:
        """ Compute expected value using value iteration

        Args:
            transition (torch.tensor): transition matrix. size=[batch_size, act_dim, state_dim, state_dim]
            reward (torch.tensor): reward matrix. size=[batch_size, act_dim, state_dim]
        
        Returns:
            q (torch.tensor): state q value. size=[horizon, batch_size, act_dim, state_dim]
        """        
        q = [torch.empty(0)] * (self.horizon)
        q[0] = reward
        for t in range(self.horizon - 1):
            v_next = torch.logsumexp(q[t], dim=-2, keepdim=True)
            q[t+1] = reward + torch.einsum("nkij, nkj -> nki", transition, v_next)
        return torch.stack(q)
    
    def plan(self, b: Tensor, value: Tensor) -> Tensor:
        """ Compute the belief action distribution 
        
        Args:
            b (torch.tensor): current belief. size=[batch_size, state_dim]
            value (torch.tensor): state q value. size=[horizon, batch_size, act_dim, state_dim]

        Returns:
            pi (torch.tensor): policy distribution. size=[batch_size, act_dim]
        """
        tau = torch.exp(self.tau.clip(math.log(1e-6), math.log(1e3)))
        tau = poisson_pdf(tau, self.horizon)
        if tau.shape[0] != b.shape[-2]:
            tau = torch.repeat_interleave(tau, b.shape[-2], 0)
        
        pi = torch.softmax(torch.einsum("...ni, h...nki -> h...nk", b, value), dim=-1)
        pi = torch.einsum("h...nk, nh -> ...nk", pi, tau)
        return pi
    
    def update_belief(self, logp_o: Tensor, a: Tensor, b: Tensor, transition: Tensor) -> Tensor:
        """ Compute state posterior
        
        Args:
            logp_o (torch.tensor): log probability of current observation. size=[batch_size, state_dim]
            a (torch.tensor): action posterior. size=[batch_size, act_dim]
            b (torch.tensor): prior belief. size=[batch_size, state_dim]
            transition (torch.tensor): transition matrix. size=[batch_size, act_dim, state_dim, state_dim]

        Returns:
            b_post (torch.tensor): state posterior. size=[batch_size, state_dim]
        """
        s_next = torch.einsum("nkij, ni, nk -> nj", transition, b, a)
        logp_s = torch.log(s_next + self.eps)
        b_post = torch.softmax(logp_s + logp_o, dim=-1)
        return b_post
    
    def init_hidden(self) -> Tensor:
        b0 = torch.softmax(self.b0, dim=-1)
        return b0
    
    def predict_one_step(self, b, u):
        transition = self.compute_transition()
        s_next = torch.einsum("...kij, ...i, ...k -> ...j", transition, b, u)
        return s_next

    def forward(
        self, logp_o: Tensor, u: Union[Tensor, None], value: Tensor,
        b: Union[Tensor, None]
        ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            logp_o (torch.tensor): sequence of observation probabilities. size=[T, batch_size, state_dim]
            u (torch.tensor): sequence of one-hot action vectors. size=[T, batch_size, act_dim]
            value (torch.tensor): precomputed q value matrix. size=[batch_size, act_dim, state_dim]
            b ([torch.tensor, None], optional): prior belief. size=[batch_size, state_dim]
        
        Returns:
            alpha_b (torch.tensor): sequence of posterior belief. size=[T, batch_size, state_dim]
            alpha_pi (torch.tensor): sequence of policy distribution. size=[T, batch_size, act_dim]
        """
        batch_size = logp_o.shape[1]
        transition = self.compute_transition()
        T = len(logp_o)

        if b is None:
            b = self.init_hidden()
            u = torch.cat([torch.ones(1, batch_size, self.act_dim).to(self.b0.device) / self.act_dim, u], dim=0)
        
        alpha_b = [b] + [torch.empty(0)] * (T) # state posterior
        alpha_pi = [torch.empty(0)] * (T) # policy
        for t in range(T):
            alpha_b[t+1] = self.update_belief(
                logp_o[t], u[t], alpha_b[t], transition
            )
            alpha_pi[t] = self.plan(alpha_b[t+1], value)
        return torch.stack(alpha_b[1:]), torch.stack(alpha_pi)


def poisson_pdf(rate: Tensor, K: int) -> Tensor:
    """ 
    Args:
        rate (torch.tensor): poission arrival rate [batch_size, 1]
        K (int): number of bins

    Returns:
        pdf (torch.tensor): truncated poisson pdf [batch_size, K]
    """
    Ks = 1 + torch.arange(K).to(rate.device)
    poisson_logp = Ks.xlogy(rate) - rate - (Ks + 1).lgamma()
    pdf = torch.softmax(poisson_logp, dim=-1)
    return pdf