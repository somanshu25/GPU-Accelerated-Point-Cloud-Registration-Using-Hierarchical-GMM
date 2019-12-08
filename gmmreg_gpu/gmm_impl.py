import argparse
import contextlib
import time

from matplotlib import mlab
import matplotlib.pyplot as plt
import numpy as np
import six

from sklearn.datasets import make_spd_matrix
from sklearn.cluster import KMeans


import cupy

eps = 1e-8

def init_gmm_params(X, k):

    kmeans = KMeans(n_clusters=k, random_state=1, max_iter=50, n_init=1).fit(X)
    means  = kmeans.cluster_centers_
    weights = np.ones((k)) / k
    #means = np.random.choice(X.flatten(), (k,X.shape[1]))
    return means, weights

@contextlib.contextmanager
def timer(message):
    cupy.cuda.Stream.null.synchronize()
    start = time.time()
    yield
    cupy.cuda.Stream.null.synchronize()
    end = time.time()
    print('%s:  %f sec' % (message, end - start))


def estimate_log_prob(X, inv_cov, means):
    xp = cupy.get_array_module(X)
    n_features = X.shape[1]
    log_det = xp.sum(xp.log(inv_cov + eps), axis=1)
    precisions = inv_cov ** 2
    log_prob = xp.sum((means ** 2 * precisions), 1) - \
        2 * xp.dot(X, (means * precisions).T) + xp.dot(X ** 2, precisions.T)
    return -0.5 * (n_features * xp.log(2 * np.pi).astype(np.float32) + log_prob) + log_det


def m_step(X, resp):
    xp = cupy.get_array_module(X)
    nk = xp.sum(resp, axis=0)
    means = xp.dot(resp.T, X) / (nk[:, None] + eps)
    X2 = xp.dot(resp.T, X * X) / (nk[:, None] + eps)
    covariances = cupy.clip(X2 - means ** 2, a_min=0.0)
    return nk / len(X), means, covariances


def e_step(X, inv_cov, means, weights):
    xp = cupy.get_array_module(X)
    weighted_log_prob = estimate_log_prob(X, inv_cov, means) + \
        xp.log(weights)
    log_prob_norm = xp.log(xp.sum(xp.exp(weighted_log_prob), axis=1) + eps)
    log_resp = weighted_log_prob - log_prob_norm[:, None]
    return xp.mean(log_prob_norm), log_resp

def train_gmm(X, max_iter, tol, means, covariances, weights=None):
    xp = cupy.get_array_module(X)
    lower_bound = -np.infty
    converged = False
    inv_cov = 1 / xp.sqrt(covariances)
    
    log_ll = []
    for n_iter in six.moves.range(max_iter):
        prev_lower_bound = lower_bound
        log_prob_norm, log_resp = e_step(X, inv_cov, means, weights)
        log_ll.append(log_prob_norm)
        weights, means, covariances = m_step(X, xp.exp(log_resp))
        inv_cov = 1 / (xp.sqrt(covariances) + eps)
        lower_bound = log_prob_norm
        change = lower_bound - prev_lower_bound
        if abs(change) < tol:
            converged = True
            break

    if not converged:
        print('Failed to converge. Increase max-iter or tol.')

    return inv_cov, means, weights, covariances, log_ll


def predict(X, inv_cov, means, weights):
    xp = cupy.get_array_module(X)
    log_prob = estimate_log_prob(X, inv_cov, means)
    return (log_prob + xp.log(weights)).argmax(axis=1)