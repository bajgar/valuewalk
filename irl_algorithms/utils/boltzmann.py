from typing import Optional

import torch


def belief_weighed_action_probs(q_vals_bac: torch.Tensor, beta: float, density_over_volume: Optional[float] = None, log_mean=False, return_logprobs=False) -> torch.Tensor:
    """
    :param q_vals_bac: (batch_size, action_dim) batch of Q values for each state-action pair
    :param beta: Boltzmann rationality coefficient
    :param density: whether to return action probabilities or action densities (treating actions as a Monte Carlo samples)
    :return: (batch_size, action_dim) batch of action probabilities
    """
    if log_mean:
        raise ValueError("This was the wrong way of calculating this. Log-mean should be taken across the batch dim, not across beliefs.")
        action_probs_bac = torch.nn.functional.log_softmax(q_vals_bac * beta, dim=-2)
        action_probs_ba = torch.mean(action_probs_bac, dim=-1)
        if density_over_volume:
            action_probs_ba = action_probs_ba + torch.log(torch.tensor(action_probs_ba.shape[-1], dtype=torch.float)) - torch.log(torch.tensor(density_over_volume, dtype=torch.float))
    elif return_logprobs:
        action_probs_ba = torch.logsumexp(beta * q_vals_bac - torch.logsumexp(beta * q_vals_bac, dim=-2, keepdim=True)
                                          , dim=-1) - torch.log(torch.tensor(q_vals_bac.shape[-1], dtype=torch.float))
    else:
        action_probs_bac = torch.nn.functional.softmax(q_vals_bac * beta, dim=-2)
        action_probs_ba = torch.mean(action_probs_bac, dim=-1)
        if density_over_volume:
            action_probs_ba = action_probs_ba * action_probs_ba.shape[-1] / density_over_volume
    return action_probs_ba
