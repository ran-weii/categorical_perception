import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

# model imports
from src.distributions.nn_models import Model, MLP
from src.algo.rl import DoubleQNetwork
from src.algo.replay_buffer import ReplayBuffer
from src.distributions.utils import kl_divergence


class DAC(Model):
    """ Discriminator actor critic for recurrent networks """
    def __init__(
        self, agent, hidden_dim, num_hidden, activation, gamma=0.9, beta=0.2, polyak=0.995, norm_obs=False,
        buffer_size=int(1e6), d_batch_size=100, a_batch_size=32, rnn_len=50, d_steps=50, a_steps=50, 
        lr_d=1e-3, lr_a=1e-3, lr_c=1e-3, decay=0, grad_clip=None, grad_penalty=1., bc_penalty=1., obs_penalty=1.
        ):
        """
        Args:
            agent (Agent): actor agent
            hidden_dim (int): value network hidden dim
            num_hidden (int): value network hidden layers
            activation (str): value network activation
            gamma (float, optional): discount factor. Default=0.9
            beta (float, optional): softmax temperature. Default=0.2
            polyak (float, optional): target network polyak averaging factor. Default=0.995
            norm_obs (bool, optional): whether to normalize observations for critic. Default=False
            buffer_size (int, optional): replay buffer size. Default=1e6
            d_batch_size (int, optional): discriminator batch size. Default=100
            a_batch_size (int, optional): actor critic batch size. Default=32
            rnn_len (int, optional): number of recurrent steps to sample. Default=50
            d_steps (int, optional): discriminator update steps per training step. Default=50
            a_steps (int, optional): actor critic update steps per training step. Default=50
            lr (float, optional): learning rate. Default=1e-3
            decay (float, optional): weight decay. Default=0
            grad_clip (float, optional): gradient clipping. Default=None
            grad_penalty (float, optional): discriminator gradient penalty. Default=1.
        """
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.polyak = polyak
        self.norm_obs = norm_obs
    
        self.buffer_size = buffer_size
        self.d_batch_size = d_batch_size
        self.a_batch_size = a_batch_size
        self.rnn_len = rnn_len
        self.d_steps = d_steps
        self.a_steps = a_steps
        self.lr = lr_d
        self.decay = decay
        self.grad_clip = grad_clip
        self.grad_penalty = grad_penalty
        self.bc_penalty = bc_penalty
        self.obs_penalty = obs_penalty

        self.agent = agent
        self.discriminator = MLP(
            input_dim=agent.obs_dim + agent.act_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            activation=activation,
            batch_norm=False
        )
        self.critic = DoubleQNetwork(
            agent.obs_dim, agent.act_dim, hidden_dim, num_hidden, activation
        )
        self.critic_target = deepcopy(self.critic)

        # freeze target parameters
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d, weight_decay=decay
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr_c, weight_decay=decay
        )
        self.actor_optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=lr_a, weight_decay=decay
        )

        self.real_buffer = ReplayBuffer(agent.obs_dim, agent.act_dim, agent.state_dim, int(1e6))
        self.replay_buffer = ReplayBuffer(agent.obs_dim, agent.act_dim, agent.state_dim, buffer_size)

        self.obs_mean = nn.Parameter(torch.zeros(agent.obs_dim), requires_grad=False)
        self.obs_variance = nn.Parameter(torch.ones(agent.obs_dim), requires_grad=False)
        
        self.plot_keys = [
            "eps_return_avg", "eps_len_avg", "d_loss_avg", 
            "critic_loss_avg", "actor_loss_avg", "obs_loss_avg"
        ]
    
    def __repr__(self):
        s_agent = self.agent.__repr__()
        s_critic = self.critic.__repr__()
        s_discriminator = self.discriminator.__repr__()
        s = "{}(gamma={}, beta={}, polyak={}, norm_obs={}, "\
            "buffer_size={}, d_batch_size={}, a_steps={}, d_steps={}, "\
            "lr={}, decay={}, grad_clip={}, grad_penalty={}"\
            "\n    discriminator={}\n    agent={}, \n    critic={}\n)".format(
            self.__class__.__name__, self.gamma, self.beta, self.polyak, self.norm_obs,
            self.replay_buffer.max_size, self.d_batch_size, self.a_steps, self.d_steps,
            self.lr, self.decay, self.grad_clip, self.grad_penalty, s_discriminator, s_agent, s_critic
        )
        return s

    def fill_real_buffer(self, dataset):
        for i in range(len(dataset)):
            batch = dataset[i]
            obs = np.array(batch["obs"])
            act = np.array(batch["act"]).reshape(-1, 1)
            rwd = np.zeros((len(obs), 1))
            done = np.zeros((len(obs), 1))
            state = np.zeros((len(obs), self.agent.state_dim))
            self.real_buffer.push(obs, act, state, rwd, done)

    def normalize_obs(self, obs):
        obs_norm = (obs - self.obs_mean) / self.obs_variance**0.5
        return obs_norm
    
    def update_normalization_stats(self):
        if self.norm_obs:
            mean = torch.from_numpy(self.replay_buffer.moving_mean).to(torch.float32).to(self.device)
            variance = torch.from_numpy(self.replay_buffer.moving_variance).to(torch.float32).to(self.device)

            self.obs_mean.data = mean
            self.obs_variance.data = variance
            
            if hasattr(self.agent, "obs_mean"):
                self.agent.obs_mean.data = mean
                self.agent.obs_variance.data = variance
    
    def reset(self):
        self.agent.reset()
    
    def choose_action(self, obs):
        obs = torch.from_numpy(obs).view(1, -1).to(torch.float32).to(self.device)
        with torch.no_grad():
            ctl = self.agent.choose_action(obs)
        return ctl.squeeze(0).numpy()
    
    def compute_reward(self, obs, ctl):
        inputs = torch.cat([obs, ctl], dim=-1)
        log_r = self.discriminator(inputs)
        r = -log_r
        return r
    
    def gradient_penalty(self, real_inputs, fake_inputs):
        # interpolate data
        alpha = torch.randn(len(real_inputs), 1).to(self.device)
        interpolated = alpha * real_inputs + (1 - alpha) * fake_inputs
        interpolated = Variable(interpolated, requires_grad=True)

        prob = torch.sigmoid(self.discriminator(interpolated))
        
        grad = torch_grad(
            outputs=prob, inputs=interpolated, 
            grad_outputs=torch.ones_like(prob),
            create_graph=True, retain_graph=True
        )[0]

        grad_norm = torch.linalg.norm(grad, dim=-1)
        return grad_norm

    def compute_discriminator_loss(self): 
        real_batch = self.real_buffer.sample_random(self.d_batch_size, prioritize=False)
        fake_batch = self.replay_buffer.sample_random(self.d_batch_size, prioritize=True)
        
        real_obs = real_batch["obs"].to(self.device)
        real_ctl = real_batch["ctl"].to(self.device)
        real_ctl = F.one_hot(real_ctl.long().squeeze(-1), self.agent.act_dim).to(torch.float32)
        fake_obs = fake_batch["obs"].to(self.device)
        fake_ctl = fake_batch["ctl"].to(self.device)
        fake_ctl = F.one_hot(fake_ctl.long().squeeze(-1), self.agent.act_dim).to(torch.float32)
        
        # normalize obs
        real_obs_norm = self.normalize_obs(real_obs)
        fake_obs_norm = self.normalize_obs(fake_obs)

        real_inputs = torch.cat([real_obs_norm, real_ctl], dim=-1)
        fake_inputs = torch.cat([fake_obs_norm, fake_ctl], dim=-1)
        inputs = torch.cat([real_inputs, fake_inputs], dim=0)

        real_labels = torch.zeros(self.d_batch_size, 1)
        fake_labels = torch.ones(self.d_batch_size, 1)
        labels = torch.cat([real_labels, fake_labels], dim=0).to(self.device)

        out = torch.sigmoid(self.discriminator(inputs))
        loss = F.binary_cross_entropy(out, labels)

        gp = self.gradient_penalty(real_inputs, fake_inputs)
        loss += self.grad_penalty * gp.mean()
        return loss

    def compute_critic_loss(self):
        batch = self.replay_buffer.sample_random(self.d_batch_size)
        obs = batch["obs"].to(self.device)
        ctl = batch["ctl"].to(self.device)
        r = batch["rwd"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        ctl_oh = F.one_hot(ctl.long().squeeze(-1), self.agent.act_dim).to(torch.float32)
        
        # normalize observation
        obs_norm = self.normalize_obs(obs)
        next_obs_norm = self.normalize_obs(next_obs)
        
        with torch.no_grad():    
            # compute reward
            r = self.compute_reward(obs_norm, ctl_oh)

            # compute value target
            q1_next, q2_next = self.critic_target(next_obs_norm)
            q_next = torch.min(q1_next, q2_next)
            v_next = torch.logsumexp(q_next / self.beta, dim=-1, keepdim=True) * self.beta
            q_target = r + (1 - done) * self.gamma * v_next
        
        q1, q2 = self.critic(obs_norm)
        q1 = torch.gather(q1, -1, ctl.long())
        q2 = torch.gather(q2, -1, ctl.long())
        q1_loss = torch.pow(q1 - q_target, 2).mean()
        q2_loss = torch.pow(q2 - q_target, 2).mean()
        q_loss = (q1_loss + q2_loss) / 2
        return q_loss

    def compute_actor_loss(self):
        batch = self.replay_buffer.sample_episodes(self.a_batch_size, self.rnn_len, prioritize=False)
        pad_batch, mask = batch
        obs = pad_batch["obs"].to(self.device)
        ctl = pad_batch["ctl"].to(self.device).to(torch.float32)
        mask = mask.to(self.device)

        # normalize observation
        obs_norm = self.normalize_obs(obs)

        [_, alpha_a], _ = self.agent(obs, ctl)
        
        q1, q2 = self.critic(obs_norm)
        q = torch.min(q1, q2)
        pi_target = torch.softmax(q / self.beta, dim=-1)
        a_loss = kl_divergence(alpha_a, pi_target)
        a_loss = torch.sum(a_loss * mask) / (mask.sum() + 1e-6)
        return a_loss
    
    def compute_obs_loss(self):
        batch = self.replay_buffer.sample_episodes(self.a_batch_size, self.rnn_len, prioritize=False)
        pad_batch, mask = batch
        obs = pad_batch["obs"].to(self.device)
        ctl = pad_batch["ctl"].to(self.device)
        
        out = self.agent(obs, ctl)
        obs_loss, _ = self.agent.obs_loss(obs, ctl, mask, out)
        obs_loss = obs_loss.mean()
        return obs_loss

    def take_gradient_step(self, logger=None):
        self.discriminator.train()
        self.critic.train()
        self.agent.train()
        self.update_normalization_stats()
        
        d_loss_epoch = []
        for i in range(self.d_steps):
            # train discriminator
            d_loss = self.compute_discriminator_loss()
            d_loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip)
            self.d_optimizer.step()
            self.d_optimizer.zero_grad()

            d_loss_epoch.append(d_loss.data.item())
            
            if logger is not None:
                logger.push({"d_loss": d_loss.data.item()})

        critic_loss_epoch = []
        actor_loss_epoch = []
        obs_loss_epoch = []
        for i in range(self.a_steps):
            # train critic
            critic_loss = self.compute_critic_loss()
            critic_loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()

            critic_loss_epoch.append(critic_loss.data.item())

            # train actor
            actor_loss = self.compute_actor_loss()
            obs_loss = self.compute_obs_loss()
            actor_total_loss = (
                actor_loss + self.obs_penalty * obs_loss
            )
            actor_total_loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_clip)
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss_epoch.append(actor_loss.data.item())
            obs_loss_epoch.append(obs_loss.data.item())
            
            # update target networks
            with torch.no_grad():
                for p, p_target in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    p_target.data.mul_(self.polyak)
                    p_target.data.add_((1 - self.polyak) * p.data)

            if logger is not None:
                logger.push({
                    "critic_loss": critic_loss.cpu().data.item(),
                    "actor_loss": actor_loss.cpu().data.item(),
                    "obs_loss": obs_loss.cpu().data.item(),
                })

        stats = {
            "d_loss": np.mean(d_loss_epoch),
            "critic_loss": np.mean(critic_loss_epoch),
            "actor_loss": np.mean(actor_loss_epoch),
            "obs_loss": np.mean(obs_loss_epoch),
        }
        
        self.discriminator.eval()
        self.critic.eval()
        self.agent.eval()
        return stats

    def on_epoch_end(self):
        pass