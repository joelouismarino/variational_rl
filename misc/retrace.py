import torch


def retrace(q_values, rewards, importance_weights=None, discount=0.9, l=0.9):
    # RETRACE: equation 3
    assert len(q_values.shape) == 2
    assert len(rewards.shape) == 2
    if q_values.shape[1] == 1:
        # degenerate case
        return q_values

    if importance_weights is None:
        # On-policy
        importance_weights = torch.ones_like(q_values)

    deltas = rewards[:, :-1] + discount*q_values[:, 1:] - q_values[:, :-1]
    importance_weights = l * torch.clamp(importance_weights, 0, 1)[:, :-2]
    importance_weights = torch.cat([l*torch.ones_like(importance_weights[:, :1]), importance_weights], 1)
    discounts = torch.cat([(discount*torch.ones_like(q_values[:, :1]))**i for i in range(q_values.shape[1])], 1)
    q_estimates = q_values[:, :1] + torch.sum(discounts[:, :-1] * torch.cumprod(importance_weights, 1) * deltas, 1, keepdim=True)

    return q_estimates


if __name__ == '__main__':
    import torch

    importance_weights = torch.rand((10, 9))
    q_values = torch.rand((10, 9))
    rewards = torch.rand((10, 9))

    print('Test degenerate case lambda is 0 and discount is 0')
    print('The target should be equal to the first q value')
    one_target = retrace(q_values, rewards, importance_weights, l=0., discount=0.)
    assert torch.all(one_target == q_values[:, :1])

    print('Test degenerate case lambda is 1 and discount is 0')
    print('The target should be equal to the first reward')
    one_target = retrace(q_values, rewards, importance_weights, l=1., discount=0.)
    assert torch.all(one_target == rewards[:, :1])

    print('Test Monte Carlo rollouts')
    print('The target should be the undiscounted sum of reward plus the value function')
    one_target = retrace(q_values, rewards, None, discount=1., l=1.)
    mc_rollouts = torch.sum(rewards[:, :-1], 1, keepdim=True) + q_values[:, -1:]
    assert torch.all(torch.isclose(one_target, mc_rollouts))

    print('Test Discounted Monte Carlo rollouts')
    print('The target should be the iscounted sum of reward plus the value function')
    one_target = retrace(q_values, rewards, None, discount=.99, l=1.)
    discounts = torch.cat([(0.99*torch.ones_like(q_values[:, :1]))**i for i in range(q_values.shape[1])], 1)
    disc_mc_rollouts = torch.sum((discounts*rewards)[:, :-1], 1, keepdim=True) + 0.99**(q_values.shape[1]-1)*q_values[:, -1:]
    assert torch.all(torch.isclose(one_target, disc_mc_rollouts))

    #print('Test changing lambda')
    #print('Should be a weighted sum of the estimators')
    #LAMBDA = 0.25
    #multiple_targets = torch.cat([retrace(q_values[:, :i+1], rewards[:, :i+1], None, discount=1., l=1.) for i in range(q_values.shape[1])], 1)
    #lambda_discounts = torch.cat([(LAMBDA*torch.ones_like(q_values[:, :1]))**(i) for i in range(q_values.shape[1])], 1)
    #lambda_target = torch.sum(lambda_discounts*multiple_targets, 1, keepdim=True)
    #one_target = retrace(q_values, rewards, None, discount=1., l=LAMBDA)
    #import pdb; pdb.set_trace()
    #assert torch.all(torch.isclose(one_target, lambda_target))


