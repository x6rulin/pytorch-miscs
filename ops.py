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

    return x
