import numpy as np
from gym.envs.classic_control import MountainCarEnv

class CustomMountainCar(MountainCarEnv):
    """ Custom mountain car environment with 
    1. discretization methods
    2. partial observation utilities """
    def __init__(self, x_bins=20, v_bins=20, x_noise=0.1, v_noise=0.01, seed=0):
        """
        Args:
            x_bins (int): number of position bins
            v_bins (int): number of velocity bins
            seed (int)
        """
        super().__init__()
        self.seed = seed
        self.x_bins = x_bins
        self.v_bins = v_bins
        self.state_dim = x_bins * v_bins
        self.act_dim = 3
        self.eps = 1e-6

        self.x_noise = x_noise
        self.v_noise = v_noise
        
        # make reward
        goal_velocity = np.linspace(self.low[1], self.high[1], self.v_bins)
        goal_position = self.goal_position * np.ones_like(goal_velocity)
        goal_obs = np.stack([goal_position, goal_velocity]).T
        goal_state = self.obs2state(goal_obs)
        self.reward = -np.ones((self.state_dim,))
        self.reward[goal_state] = 0

    def batch_step(self, state, action):
        """ Batch apply dynamics 
        
        Args:
            state (np.array): [batch_size, 2]
            action (np.array): [batch_size]
        """
        assert len(list(action.shape)) == 1
        position = state[:, 0].copy()
        velocity = state[:, 1].copy()
        velocity += (action - 1) * self.force + np.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        
        # handle min position
        is_invalid = np.stack([position <= self.min_position, velocity < 0]).T
        is_valid = np.all(is_invalid, axis=1) == False
        velocity *= is_valid
        
        next_state = np.stack([position, velocity]).T
        return next_state
    
    def obs2state(self, obs):
        obs = obs.reshape(-1, 2).copy()
        d_x = (self.high[0] - self.low[0]) / self.x_bins
        d_v = (self.high[1] - self.low[1]) / self.v_bins
        x_grid = np.clip((obs[:, 0] - self.low[0]) // d_x, 0, self.x_bins - 1)
        v_grid = np.clip((obs[:, 1] - self.low[1]) // d_v, 0, self.v_bins - 1)
        state = x_grid + v_grid * self.x_bins
        return state.astype(int)

    def state2obs(self, state):
        assert np.all(state <= self.state_dim)
        d_x = (self.high[0] - self.low[0]) / self.x_bins
        d_v = (self.high[1] - self.low[1]) / self.v_bins

        v_grid = np.floor(state / self.x_bins)
        x_grid = state - v_grid * self.x_bins

        position = x_grid * d_x + self.low[0]
        velocity = v_grid * d_v + self.low[1]

        obs = np.stack([position, velocity]).T
        return obs
    
    def make_initial_distribution(self, num_samples=200):
        """ Create initial state distribution """
        np.random.seed(self.seed)
        position = np.random.uniform(-0.6, -0.4, num_samples)
        velocity = np.zeros((num_samples,))
        obs = np.stack([position, velocity]).T
        state = self.obs2state(obs)

        initial_dist = self.eps * np.ones((self.state_dim,))
        unique_state, counts = np.unique(state, return_counts=True)
        initial_dist[unique_state] += counts
        initial_dist /= initial_dist.sum(keepdims=True)
        self.initial_dist = initial_dist

    def make_transition_matrix(self, num_samples=8000):
        """ Create discrete transition marix """
        np.random.seed(self.seed)

        # sample the observation space
        position = np.random.uniform(self.low[0], self.high[0], num_samples)
        velocity = np.random.uniform(self.low[1], self.high[1], num_samples)
        obs = np.stack([position, velocity]).T
        state = self.obs2state(obs)
        
        transition_matrix = self.eps * np.ones((self.act_dim, self.state_dim, self.state_dim))
        for a in range(self.act_dim):
            action = a * np.ones((num_samples,))
            next_obs = self.batch_step(obs, action)
            next_state = self.obs2state(next_obs)
            for s in range(self.state_dim):
                unique_next_state, counts = np.unique(
                    next_state[state == s], return_counts=True
                )
                transition_matrix[a, s, unique_next_state] += counts

        transition_matrix /= transition_matrix.sum(-1, keepdims=True)
        self.transition_matrix = transition_matrix

    def make_observation_matrix(self, num_samples=8000):
        """ Create discrete transition matrix """
        np.random.seed(self.seed)
        # sample the observation space
        position = np.random.uniform(self.low[0], self.high[0], num_samples)
        velocity = np.random.uniform(self.low[1], self.high[1], num_samples)
        obs = np.stack([position, velocity]).T
        state = self.obs2state(obs)

        # sample observations
        x_noise = self.x_noise * np.random.normal(
            np.zeros((num_samples,)), np.ones((num_samples,))
        )
        v_noise = self.v_noise * np.random.normal(
            np.zeros((num_samples,)), np.ones((num_samples,))
        )

        next_position = position + x_noise
        next_velocity = velocity + v_noise
        next_obs = np.stack([next_position, next_velocity]).T
        next_state = self.obs2state(next_obs)

        obs_matrix = self.eps * np.ones((self.state_dim, self.state_dim))
        for s in range(self.state_dim):
            unique_next_state, counts = np.unique(
                next_state[state == s], return_counts=True
            )
            obs_matrix[s, unique_next_state] += counts
        
        obs_matrix /= obs_matrix.sum(-1, keepdims=True)
        self.obs_matrix = obs_matrix