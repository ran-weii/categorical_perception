import time
import torch

def value_iteration(
    transition_matrix, reward, gamma, softmax=True, 
    alpha=1., max_iter=100, tol=1e-5
    ):
    """
    Args:
        transition_matrix (torch.tensor): transition matrix [act_dim, state_dim, state_dim]
        reward (torch.tensor): reward vector [state_dim]
        gamma (float): discount factor
        softmax (bool): whether to use soft value iteration. Default=True
        alpha (float): softmax temperature
        max_iter (int): max iteration. Default=100
        tol (float): error tolerance. Default=1e-5

    Returns:
        q (torch.tensor): q function [state_dim, act_dim]
        info (dict): {"tol", "iter"}
    """
    start = time.time()
    state_dim = transition_matrix.shape[1]
    act_dim = transition_matrix.shape[0]
    
    q = [torch.zeros(state_dim, act_dim)]
    for t in range(max_iter):
        if softmax:
            v = torch.logsumexp(alpha * q[t], dim=-1) / alpha
        else:
            v = q[t].max(-1)[0]
        q_t = torch.sum(transition_matrix * (reward + gamma * v).view(1, 1, -1), dim=-1).T
        q.append(q_t)

        q_error = torch.abs(q_t - q[t]).mean()
        if q_error < tol:
            break
    
    tnow = time.time() - start
    return q[-1], {"tol": q_error.item(), "iter": t, "time": tnow}