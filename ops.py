"""Commonly used options implemented by Pytorch, out of the box. """
import numpy as np
import torch


def truncated_normal(uniform):
    return parameterized_truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2)


def sample_truncated_normal(shape=()):
    return truncated_normal(torch.from_numpy(np.random.uniform(0, 1, shape)))


def parameterized_truncated_normal(uniform, mu, sigma, a, b):
    r"""Implements the sampling of truncated normal distribution using the inversed
        cumulative distribution function (CDF) method.

    .. _Truncated Normal\: normal distribution in which the range of definition is
                           made finite at one or both ends of the interval.
        https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """
    normal = torch.distributions.normal.Normal(0, 1)

    alpha, beta = (a - mu) / sigma, (b - mu) / sigma
    p = normal.cdf(alpha) + (normal.cdf(beta) - normal.cdf(alpha)) * uniform
    p = p.numpy()

    one = np.array(1, dtype=p.dtype)
    epsilon = np.array(np.finfo(p.dtype).eps, dtype=p.dtype)
    v = np.clip(2 * p - 1, -one + epsilon, one - epsilon)
    x = mu + sigma * np.sqrt(2) * torch.erfinv(torch.from_numpy(v))
    x = torch.clamp(x, a, b)

    return x.type(torch.get_default_dtype())


class FocalLoss(torch.nn.Module):
    r"""This criterion is a implemenation of Focal Loss, which is proposed in Focal Loss for
        Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable): the scalar factor for this criterion
            gamma(float, double): gamma > 0; reduces the relative loss for well-classiﬁed examples
                                  (p > .5), putting more focus on hard, misclassified examples
            size_average(bool): By default, the losses are averaged over observations for each
                                minibatch. However, if the field size_average is set to False, the
                                losses are instead summed for each minibatch.
    """
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        if isinstance(alpha, (float, int)):
            alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            alpha = torch.Tensor(alpha)
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.reshape(*input.shape[:2], -1)  # N, C, H, W => N, C, H*W
            input = input.transpose(1, 2)                # N, C, H*W => N, H*W, C
            input = input.reshape(-1, input.size(2))     # N, H*W, C => N*H*W, C
        target = target.reshape(-1, 1)

        logpt = torch.nn.functional.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.reshape(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.reshape(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        return (self.size_average and loss.mean()) or loss.sum()
