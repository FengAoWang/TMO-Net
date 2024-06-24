import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from lifelines.utils import concordance_index
from torch.distributions import Normal, kl_divergence


def KL_loss(mu, logvar, beta, c=0.0):
    # KL divergence loss
    KLD_1 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return beta * KLD_1


def KL_divergence(mu1, mu2, log_sigma1, log_sigma2):
    p = Normal(mu1, torch.exp(log_sigma1))
    q = Normal(mu2, torch.exp(log_sigma2))

    # 计算KL损失
    kl_loss = kl_divergence(p, q).mean()
    return kl_loss


def reconstruction_loss(recon_x, x, recon_param, dist):
    batch_size = x.size(0)
    if dist == 'bernoulli':
        BCE = nn.BCEWithLogitsLoss(reduction='sum')
        recons_loss = BCE(recon_x, x) / batch_size
    elif dist == 'gaussian':
        mse = nn.MSELoss(reduction='sum')
        recons_loss = mse(recon_x, x) / batch_size
    elif dist == 'F2norm':
        recons_loss = torch.norm(recon_x-x, p=2)
    elif dist == 'prob':
        recons_loss = recon_x.log_prob(x).sum(dim=1).mean(dim=0)
    elif dist == 'Poisson':
        pois_loss = nn.PoissonNLLLoss(full=True, reduction='sum')
        recons_loss = pois_loss(recon_x, x)
    elif dist == 'ce':
        x = torch.argmax(x, dim=1)
        ce_loss = nn.CrossEntropyLoss(reduction='sum')
        recons_loss = ce_loss(recon_x, x)
    else:
        raise AttributeError("invalid dist")

    return recon_param * recons_loss


def r_set(x):
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)
    return indicator_matrix


def c_index(pred, ytime, yevent):
    n_sample = len(ytime)
    ytime_indicator = r_set(ytime)
    ytime_matrix = ytime_indicator - torch.diag(torch.diag(ytime_indicator))
    censor_idx = (yevent == 0).nonzero()
    zeros = torch.zeros(n_sample)
    ytime_matrix[censor_idx, :] = zeros
    pred_matrix = torch.zeros_like(ytime_matrix)
    for j in range(n_sample):
        for i in range(n_sample):
            if pred[i] < pred[j]:
                pred_matrix[j, i] = 1
            elif pred[i] == pred[j]:
                pred_matrix[j, i] = 0.5

    concord_matrix = pred_matrix.mul(ytime_matrix)
    concord = torch.sum(concord_matrix)
    epsilon = torch.sum(ytime_matrix)
    concordance_index = torch.div(concord, epsilon)
    if torch.cuda.is_available():
        concordance_index = concordance_index.cuda()
    return concordance_index


def cox_loss(survtime, censor, hazard_pred):
    current_batch_len = len(survtime)

    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).cuda()
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox
