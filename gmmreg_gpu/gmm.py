from __future__ import print_function
from __future__ import division
import abc
import six
import numpy as np
import open3d as o3
from sklearn import mixture

from gmm_impl import train_gmm, init_gmm_params, timer
import cupy

@six.add_metaclass(abc.ABCMeta)
class Feature():
	@abc.abstractmethod
	def init(self):
		pass

	@abc.abstractmethod
	def compute(self, data):
		return None

	def annealing(self):
		pass

	def __call__(self, data):
		return self.compute(data)
		
class GMM_Sklearn(Feature):
	def __init__(self, n_gmm_components=50):
		self._n_gmm_components = n_gmm_components

	def init(self):
		self._clf = mixture.GaussianMixture(n_components=self._n_gmm_components, max_iter=30,
											covariance_type='diag')

	def compute(self, data):
		self._clf.fit(data)
		return self._clf.means_, self._clf.weights_

class GMM_GPU(Feature):
	def __init__(self, n_gmm_components=100, max_iter=30, tol=1e-4):
		self._n_gmm_components = n_gmm_components
		self.max_iter = max_iter
		self.tol = tol

	def init(self):
		self._clf = GMM_GPU_Base(self._n_gmm_components, max_iter=self.max_iter, tol=self.tol)

	def compute(self, data):
		self._clf.fit(data)
		return self._clf.means_, self._clf.weights_
	
class GMM_CPU(Feature):
	def __init__(self, n_gmm_components=100, max_iter=30, tol=1e-4):
		self._n_gmm_components = n_gmm_components
		self.max_iter = max_iter
		self.tol = tol		

	def init(self):
		self._clf = GMM_CPU_Base(self._n_gmm_components, max_iter=self.max_iter, tol=self.tol)

	def compute(self, data):
		self._clf.fit(data)
		return self._clf.means_, self._clf.weights_

class GMM_GPU_Base:
    def __init__(self, num_components, max_iter=30, tol=1e-4):
        self.num_components = num_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        means, weights = init_gmm_params(X, self.num_components)
        covs = 0.1*np.ones((self.num_components, means.shape[1]), dtype=np.float32)
        
        dev_X = cupy.asarray(X)
        dev_means = cupy.asarray(means)
        dev_covs = cupy.asarray(covs)
        dev_weights = cupy.asarray(weights)
        
        with timer('GPU GMM TRAIN'):
            inv_covs, dev_means, dev_weights, dev_covs, lls = train_gmm(dev_X, self.max_iter, self.tol, dev_means, dev_covs, dev_weights)
                
        means, covs, weights = cupy.asnumpy(dev_means), cupy.asnumpy(dev_covs), cupy.asnumpy(dev_weights)

        self.means_ = means
        self.covariances_ = covs
        self.weights_ = weights
        self.lls = lls
        self.inv_covs = inv_covs

        return self
    
    def predict(self, X):
        return predict(X, self.inv_covs, self.means, self.weights)

class GMM_CPU_Base:
    def __init__(self, num_components, max_iter=30, tol=1e-4):
        self.num_components = num_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        means, _, weights = init_gmm_params(X, self.num_components)
        covs = 0.1*np.ones((self.num_components, means.shape[1]), dtype=np.float32)
        
        dev_X = X
        dev_means = means
        dev_covs = covs
        dev_weights = weights

        with timer('CPU GMM TRAIN'):
            inv_covs, dev_means, dev_weights, dev_covs, lls = train_gmm(dev_X, self.max_iter, self.tol, dev_means, dev_covs, dev_weights)
                
        means, covs, weights = cupy.asnumpy(dev_means), cupy.asnumpy(dev_covs), cupy.asnumpy(dev_weights)

        self.means_ = means
        self.covariances_ = covs
        self.weights_ = weights
        self.lls = lls
        self.inv_covs = inv_covs

        return self
    
    def predict(self, X):
        return predict(X, self.inv_covs, self.means, self.weights)