import math
import torch


class Helen(torch.optim.Optimizer):
    """ Helen optimizer.
    :param embed_params: iterable of parameters to optimize or dicts defining
        parameter groups
    :param net_params: iterable of parameters to optimize or dicts defining
        parameter groups
    :param lr_embed: learning rate for embedding parameters (default: 1e-3)
    :param lr_net: learning rate for network parameters (default: 1e-3)
    :param rho: scaling factor (default: 0.05)
    :param net_pert: whether to perturb dense network parameters (default: True)
    :param bound: bound for frequency scaling (default: 0.3)
    :param betas: coefficients used for computing running averages of gradient
        and its square (default: (0.9, 0.999))
    :param eps: term added to the denominator to improve numerical stability
        (default: 1e-8)
    :param weight_decay: weight decay (L2 penalty) (default: 0)
    :param adaptive: whether to use adaptive scaling (default: False)

    """

    def __init__(self, embed_params, net_params, lr_embed=1e-3, lr_net=1e-3, rho=0.05, net_pert=True, bound=0.3,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0, adaptive=False, **kwargs):
        if not 0.0 <= lr_net or not 0.0 <= lr_embed:
            raise ValueError("Invalid learning rate: {}".format(lr_embed))
        if not 0.0 <= rho:
            raise ValueError("Invalid rho value: {}".format(rho))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(rho=rho, bound=bound, betas=betas, eps=eps, weight_decay=weight_decay, adaptive=adaptive)
        self.net_pert = net_pert
        param_group = [
            {'params': embed_params, 'lr': lr_embed, 'embed': True},
            {'params': net_params, 'lr': lr_net, 'embed': False}
        ]
        super().__init__(param_group, defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue

                if group['embed']:
                    grad_norm = torch.norm(p.grad)
                    scale = group["rho"] / (grad_norm + 1e-12)

                    self.state[p]["old_p"] = p.data.clone()
                    unique_ids = self.state[p]["unique_ids"]
                    unique_ids_count = self.state[p]["unique_ids_count"].float()

                    freq_scale = torch.scatter(torch.zeros(p.shape[0], device=p.device), 0, unique_ids,
                                               unique_ids_count)

                    freq_scale = torch.clamp(freq_scale / torch.max(freq_scale), group["bound"])
                    e_w = p.grad * scale.to(p)
                    e_w = e_w * freq_scale.unsqueeze(1)
                    p.add_(e_w)
                elif self.net_pert:
                    scale = group["rho"] / (grad_norm + 1e-12)
                    self.state[p]["old_p"] = p.data.clone()
                    e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                    p.add_(e_w)
                else:
                    self.state[p]["old_g"] = p.grad.clone()

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None: continue

                if group['embed'] or self.net_pert:
                    p.data = self.state[p]["old_p"]
                else:
                    p.grad = self.state[p]["old_g"]

                grad = p.grad.data
                state = self.state[p]
                # State initialization
                if 'exp_avg' not in state:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                step_size = group['lr'] / bias_correction1
                adam_step = exp_avg / (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add(group['eps'])
                p.data.add_(adam_step, alpha=-step_size)

        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "step func requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def count_feature_occurrence(self, X, feature_params_map, feature_specs):
        """
        Count the occurrence of each feature in the batch
        add result to state of optimizer
        :param X: batch data
        :param feature_params_map: map from feature name to its parameters
        :param feature_specs: feature specs
        """
        X = X.long()
        for feature, feature_spec in feature_specs.items():
            feature_idx = feature_spec["index"]
            feature_field_batch = X[:, feature_idx]
            unique_features, feature_count = torch.unique(feature_field_batch, return_counts=True, sorted=True)

            feature_params = feature_params_map[feature]
            for param in feature_params:
                self.state[param]["unique_ids"] = unique_features
                self.state[param]["unique_ids_count"] = feature_count
