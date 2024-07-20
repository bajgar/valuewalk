import torch


def kl_between_gaussians(mean1, std1, mean2, std2):
    """
    Compute the KL divergence between two Gaussians with diagonal covariance matrices
    :param mean1: the mean of the first Gaussian
    :param std1: the standard deviation of the first Gaussian
    :param mean2: the mean of the second Gaussian
    :param std2: the standard deviation of the second Gaussian
    :return: the KL divergence from the first to the second Gaussian
    """
    log_std1 = torch.log(std1)
    log_std2 = torch.log(std2)
    return log_std2 - log_std1 + (std1 ** 2 + (mean1 - mean2) ** 2) * 0.5 / std2 ** 2 - 0.5
