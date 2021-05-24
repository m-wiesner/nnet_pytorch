import torch
import random
from torch.optim.optimizer import Optimizer, required


class AcceleratedSGLD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """
    @staticmethod
    def add_args(parser):
        parser.add_argument('--sgld-stepsize', type=float, default=1.0)
        parser.add_argument('--sgld-noise', type=float, default=0.001)
        parser.add_argument('--sgld-replay-correction', type=float, default=0.5)
        parser.add_argument('--sgld-weight-decay', type=float, default=0.0)
        parser.add_argument('--sgld-epsilon', type=float, default=5e-05)
        parser.add_argument('--sgld-overshoot', type=float, default=0.0)

    @classmethod
    def build_partial(cls, conf):
        return partial(
            SGLD.__init__,
            lr=conf['sgld_stepsize'],
            noise=conf['sgld_noise'],
            stepscale=conf['sgld_replay_correction'],
            weight_decay=conf['sgld_weight_deccay'],
            rel_overshoot=conf['sgld_overshoot'],
            epsilon=conf['sgld-epsilon'],
        )

    def __init__(self, params, finalval, lr=required, momentum=0, dampening=0,
        weight_decay=0, nesterov=False, stepscale=1.0, noise=0.005,
        rel_overshoot=0.1, epsilon=0.00005,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(AcceleratedSGLD, self).__init__(params, defaults)
        self.noise = noise
        self.stepscale = stepscale
        # Shoot for 10% better (helps with gradient)
        if finalval >= 0:
            self.final_val = finalval * (1 - rel_overshoot)
        else:
            self.final_val = finalval * (1 + rel_overshoot)
        self.epsilon = epsilon

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def langevin_noise(self, x, std=1.0):
        return self.noise * torch.randn_like(x).mul_(std)
    
    @torch.no_grad()
    def step(self, startval=None, numsteps=None, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad_norm = max(self.epsilon, (p.grad.data ** 2.0).sum())
                if grad_norm <= self.epsilon:
                    print("Small Grad Norm!!")
                    grad_norm = self.epsilon
                opt_lr = (self.final_val - startval)/grad_norm
                # When we are below the requested value, we can just descend at
                # at a normal pace ...
                opt_lr = self.epsilon / grad_norm if opt_lr > 0 else -opt_lr
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                
                replay_correction = numsteps[:, None, None] ** self.stepscale
                langevin_std = 1.0 / replay_correction
                
                self.state[p]['update'] = self.langevin_noise(p.data, std=langevin_std).add_(
                    d_p.div_(replay_correction),
                    alpha=-group['lr'] * opt_lr,
                )
                p.data.add_(self.state[p]['update'])
                self.state[p]['opt_lr'] = opt_lr 
        return loss
