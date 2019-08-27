import torch


def retrace(q_values, rewards, importance_weights=None, discount=0.9, l=0.9):
    # RETRACE: equation 3
    assert len(q_values.shape) == 2
    assert len(rewards.shape) == 2
    if q_values.shape[0] == 1:
        # degenerate case
        return q_values

    if importance_weights is None:
        # On-policy
        importance_weights = torch.ones_like(q_values)

    deltas = q_values[:, :-1] - (rewards[:, 1:] + discount*q_values[:, 1:])
    importance_weights = discount * l * torch.clamp(importance_weights, 0, 1)[:, :-2]
    importance_weights = torch.cat([torch.ones_like(importance_weights[:, :1]), importance_weights], 1)
    q_estimates = q_values[:, :1] + torch.sum(torch.cumprod(importance_weights, 1) * deltas, 1, keepdim=True)

    return q_estimates


if __name__ == '__main__':
    import torch

    importance_weights = torch.rand((10, 9))
    q_values = torch.rand((10, 9))
    rewards = torch.rand((10, 9))
    one_target = retrace(q_values, rewards, importance_weights)
    print(one_target)
    multiple_targets = torch.cat([retrace(q_values[:, i:], rewards[:, i:], importance_weights[:, i:]) for i in range(q_values.shape[1])], 1)
    print(multiple_targets)


