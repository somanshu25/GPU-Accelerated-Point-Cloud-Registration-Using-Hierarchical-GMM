import argparse
import contextlib
import time

from matplotlib import mlab
import matplotlib.pyplot as plt
import six
import numpy

from sklearn.datasets import make_spd_matrix
from sklearn.cluster import KMeans


import cupy
from cupy import float32
eps = 1e-8

def row_norms(X, squared=False):
	xp = cupy.get_array_module(X)
	norms = xp.einsum('ij,ij->i', X, X)

	if not squared:
		xp.sqrt(norms, norms)
	return norms    

def init_gmm_params(X, k, cov_type='diag'):
	
	#kmeans = KMeans(n_clusters=k, random_state=2, max_iter=50, n_init=1).fit(X)
	#means  = kmeans.cluster_centers_

	xp = cupy.get_array_module(X)

	weights = xp.ones(k, dtype=float32) / k
	means = xp.random.choice(X.flatten(), (k,X.shape[1]))

	if cov_type == 'diag':
		covs = 0.1*xp.ones((k, X.shape[1]), dtype=float32)
	elif cov_type == 'spherical':
		covs = 0.1*xp.ones((k,), dtype=float32)

	return means.astype(float32), weights, covs

@contextlib.contextmanager
def timer(message):
	cupy.cuda.Stream.null.synchronize()
	start = time.time()
	yield
	cupy.cuda.Stream.null.synchronize()
	end = time.time()
	print('%s:  %f sec' % (message, end - start))


def estimate_log_prob_spherical(X, inv_cov, means):
	xp = cupy.get_array_module(X)
	n_features = X.shape[1]

	log_det = n_features * (xp.log(inv_cov + eps))

	precisions = inv_cov ** 2

	log_prob = (xp.sum(means ** 2, 1) * precisions -
				2 * xp.dot(X, means.T * precisions) +
				xp.outer(row_norms(X, squared=True), precisions))

	return -0.5 * (n_features * xp.log(2 * xp.pi).astype(xp.float32) + log_prob) + log_det

def estimate_log_prob(X, inv_cov, means):
	xp = cupy.get_array_module(X)
	n_features = X.shape[1]

	log_det = xp.sum(xp.log(inv_cov + eps), axis=1)

	precisions = inv_cov ** 2

	log_prob = xp.sum((means ** 2 * precisions), 1) - \
		2 * xp.dot(X, (means * precisions).T) + xp.dot(X ** 2, precisions.T)

	return -0.5 * (n_features * xp.log(2 * xp.pi).astype(xp.float32) + log_prob) + log_det


def estimate_covariance(resp, X, nk, means, reg_covar=1e-6):
	xp = cupy.get_array_module(X)

	avg_X2 = xp.dot(resp.T, X * X) / nk[:, None]
	avg_means2 = means ** 2
	avg_X_means = means * cupy.dot(resp.T, X) / nk[:, None]

	return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar    

def m_step(X, resp, cov_type='diag'):
	xp = cupy.get_array_module(X)
	nk = xp.sum(resp, axis=0) + eps
	means = xp.dot(resp.T, X) / (nk[:, None])

	#X2 = xp.dot(resp.T, X * X) / (nk[:, None])
	#covariances = cupy.clip(X2 - means ** 2, a_min=1e-6)

	covariances = estimate_covariance(resp, X, nk, means)

	if cov_type == 'spherical':
		covariances = xp.mean(covariances, axis=1)

	return nk / len(X), means, covariances

def e_step(X, inv_cov, means, weights, cov_type='diag'):
	xp = cupy.get_array_module(X)

	if cov_type == 'diag':
		weighted_log_prob = estimate_log_prob(X, inv_cov, means) + xp.log(weights + eps)
	elif cov_type == 'spherical':
		weighted_log_prob = estimate_log_prob_spherical(X, inv_cov, means) + xp.log(weights + eps)

	log_prob_norm = xp.log(xp.sum(xp.exp(weighted_log_prob), axis=1) + eps)
	log_resp = weighted_log_prob - log_prob_norm[:, None]

	return xp.mean(log_prob_norm), log_resp

def train_gmm(X, max_iter, tol, means, covariances, weights, cov_type='diag'):
	xp = cupy.get_array_module(X)
	lower_bound = -numpy.infty
	converged = False
	inv_cov = 1 / xp.sqrt(covariances)
	
	log_ll = []
	for n_iter in six.moves.range(max_iter):
		prev_lower_bound = lower_bound

		log_prob_norm, log_resp = e_step(X, inv_cov, means, weights, cov_type)

		log_ll.append(log_prob_norm)

		weights, means, covariances = m_step(X, xp.exp(log_resp), cov_type)
		#print(covariances)
		inv_cov = 1 / (xp.sqrt(covariances + 1e-6) + eps)

		lower_bound = log_prob_norm
		change = lower_bound - prev_lower_bound
		if abs(change) < tol:
			converged = True
			break

	if not converged:
		print('Failed to converge. Increase max-iter or tol.')

	return inv_cov, means, weights, covariances, log_ll

def predict(X, inv_cov, means, weights, cov_type='diag'):
	xp = cupy.get_array_module(X)

	if cov_type == 'diag':
		log_prob = estimate_log_prob(X, inv_cov, means)
	elif cov_type == 'spherical':
		log_prob = estimate_log_prob_spherical(X, inv_cov, means)

	return (log_prob + xp.log(weights + eps)).argmax(axis=1)