import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torch_dist
from src.agents.qmdp_layer import QMDPLayer
from src.distributions.mixture_models import ConditionalGaussian
from src.distributions.utils import kl_divergence

from typing import Union, Tuple, Optional
from torch import Tensor

class VINAgent(nn.Module):
    """ Value iteraction network agent with 
    conditinal gaussian observation and discrete control and 
    QMDP hidden layer
    """
    def __init__(
        self, state_dim, act_dim, obs_dim, rank, horizon,
        obs_cov="full"
        ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.horizon = horizon
        
        self.rnn = QMDPLayer(state_dim, act_dim, rank, horizon)
        self.obs_model = ConditionalGaussian(
            obs_dim, state_dim, cov=obs_cov, batch_norm=True
        )
        self.c = nn.Parameter(torch.randn(1, state_dim))
        self._pi0 = nn.Parameter(torch.randn(1, act_dim, state_dim))
        nn.init.xavier_normal_(self.c, gain=1.)
        nn.init.xavier_normal_(self._pi0, gain=1.)
    
    def reset(self):
        """ Reset internal states for online inference """
        self._b = None # torch.ones(1, self.state_dim)
        self._a = None # previous action distribution
        self._prev_ctl = None # size=[1, batch_size, act_dim]

    @property
    def target_dist(self):
        return torch.softmax(self.c, dim=-1)
    
    @property
    def pi0(self):
        """ Prior policy """
        return torch.softmax(self._pi0, dim=-1)
    
    @property
    def transition(self):
        return self.rnn.transition
    
    @property
    def value(self):
        value = self.rnn.compute_value(self.transition, self.reward)
        return value
    
    @property
    def policy(self):
        """ Optimal planned policy """
        b = torch.eye(self.state_dim)
        pi = self.rnn.plan(b, self.value)
        return pi
    
    @property
    def passive_dynamics(self):
        """ Optimal controlled dynamics """
        policy = self.policy.T.unsqueeze(-1)
        transition = self.transition.squeeze(0)
        return torch.sum(transition * policy, dim=-3)
    
    @property
    def efe(self):
        """ Negative expected free energy """
        entropy = self.obs_model.entropy()
        c = self.target_dist
        kl = kl_divergence(torch.eye(self.state_dim), c)
        return -kl - entropy
    
    @property
    def reward(self):
        """ State action reward """
        transition = self.rnn.transition
        entropy = self.obs_model.entropy()
        
        c = self.target_dist
        kl = kl_divergence(transition, c.unsqueeze(-2).unsqueeze(-2))
        eh = torch.sum(transition * entropy.unsqueeze(-2).unsqueeze(-2), dim=-1)
        log_pi0 = torch.log(self.pi0 + 1e-6)
        r = -kl - eh + log_pi0
        return r

    def forward(
        self, o: Tensor, u: Union[Tensor, None], 
        hidden: Optional[Union[Tuple[Tensor, Tensor], None]]=None
        ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """ 
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
            hidden ([tuple[torch.tensor, torch.tensor], None], optional). initial hidden state.
        
        Returns:
            alpha_b (torch.tensor): state belief distributions. size=[T, batch_size, state_dim]
            alpha_a (torch.tensor): action predictive distributions. size=[T, batch_size, act_dim]
        """
        b, a = None, None
        if hidden is not None:
            b, a = hidden

        logp_o = self.obs_model.log_prob(o) 
        reward = self.reward
        alpha_b, alpha_a = self.rnn(logp_o, u, reward, b, a)
        return [alpha_b, alpha_a], [alpha_b, alpha_a] # second tuple used in bptt
    
    def act_loss(self, o, u, mask, forward_out):
        """ Compute action loss 
        
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
            mask (torch.tensor): binary mask sequence. size=[T, batch_size]
            forward_out (list): outputs of forward method

        Returns:
            loss (torch.tensor): action loss. size=[batch_size]
            stats (dict): action loss stats
        """
        _, alpha_a = forward_out
        logp_u = self.ctl_model.mixture_log_prob(alpha_a, u)
        loss = -torch.sum(logp_u * mask, dim=0) / (mask.sum(0) + 1e-6)

        # compute stats
        nan_mask = mask.clone()
        nan_mask[nan_mask == 0] = torch.nan
        logp_u_mean = -torch.nanmean((nan_mask * logp_u)).cpu().data
        stats = {"loss_u": logp_u_mean}
        return loss, stats
    
    def obs_loss(self, o, u, mask, forward_out):
        """ Compute observation loss 
        
        Args:
            o (torch.tensor): observation sequence. size=[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size=[T, batch_size, ctl_dim]
            mask (torch.tensor): binary mask tensor. size=[T, batch_size]
            forward_out (list): outputs of forward method

        Returns:
            loss (torch.tensor): observation loss. size=[batch_size]
            stats (dict): observation loss stats
        """
        alpha_b, _ = forward_out
        logp_o = self.obs_model.mixture_log_prob(alpha_b, o)
        loss = -torch.sum(logp_o * mask, dim=0) / (mask.sum(0) + 1e-6)

        # compute stats
        nan_mask = mask.clone()
        nan_mask[nan_mask == 0] = torch.nan
        logp_o_mean = -torch.nanmean((nan_mask * logp_o)).cpu().data
        stats = {"loss_o": logp_o_mean}
        return loss, stats
    
    def choose_action(self, o):
        """ Choose action online for a single time step
        
        Args:
            o (torch.tensor): observation sequence. size[batch_size, obs_dim]
            u (torch.tensor): control sequence. size[batch_size, ctl_dim]
            sample_method (str, optional): sampling method. 
                choices=["bma", "ace", "ace"]. Default="ace"
            num_samples (int, optional): number of samples to draw. Default=1
        
        Returns:
            u_sample (torch.tensor): sampled controls. size=[num_samples, batch_size]
            logp (torch.tensor): control log probability. size=[num_samples, batch_size]
        """
        [alpha_b, alpha_a], _ = self.forward(
            o.unsqueeze(0), self._prev_ctl, [self._b, self._a]
        )
        b_t, a_t = alpha_b[0], alpha_a[0]
        u_sample = torch_dist.Categorical(a_t).sample()
        
        self._b, self._a = b_t, a_t
        self._prev_ctl = F.one_hot(u_sample, num_classes=self.act_dim).unsqueeze(0).to(torch.float32)
        return u_sample
    
    def choose_action_batch(self, o, u):
        """ Choose action offline for a batch of sequences 
        
        Args:
            o (torch.tensor): observation sequence. size[T, batch_size, obs_dim]
            u (torch.tensor): control sequence. size[T, batch_size, ctl_dim]
            sample_method (str, optional): sampling method. 
                choices=["bma", "ace", "ace"]. Default="ace"
            num_samples (int, optional): number of samples to draw. Default=1
            tau (float, optional): gumbel softmax temperature. Default=0.1
            hard (bool, optional): if hard use straight-through gradient. Default=True
            return_hidden (bool, optional): if true return agent hidden state. Default=False

        Returns:
            u_sample (torch.tensor): sampled controls. size=[num_samples, T, batch_size]
            logp (torch.tensor): control log probability. size=[num_samples, T, batch_size]
        """
        [alpha_b, alpha_a], _ = self.forward(o, u)
        
        u_sample = torch_dist.Categorical(alpha_a).sample()
        return u_sample

    def predict(self, o, u, sample_method="ace", num_samples=1):
        """ Offline prediction observations """
        [alpha_b, alpha_a], _ = self.forward(o, u)

        if sample_method == "bma":
            o_sample = self.obs_model.bayesian_average(alpha_b)
        else:
            sample_mean = True if sample_method == "acm" else False
            o_sample = self.obs_model.ancestral_sample(
                alpha_b, num_samples, sample_mean, tau=0.1, hard=True
            )
        return o_sample

if __name__ == "__main__":
    state_dim = 10
    act_dim = 5
    obs_dim = 12
    rank = 7
    horizon = 9
    agent = VINAgent(state_dim, act_dim, obs_dim, rank, horizon)
    
    # synthetic data
    T = 15
    batch_size = 32
    o = torch.randn(T, batch_size, obs_dim)
    u = torch.softmax(torch.randn(T, batch_size, act_dim), dim=-1)

    # test forward
    [b, a], _ = agent(o, u)
    
    # test batch inference
    agent.choose_action_batch(o, u)

    # test control
    agent.reset()
    u = agent.choose_action(o[0])
    u = agent.choose_action(o[1])