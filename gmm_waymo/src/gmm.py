from __future__ import print_function
from __future__ import division
import abc
import six
import numpy as np
import open3d as o3
from sklearn import mixture, svm
import time
from gmm_impl import train_gmm, init_gmm_params, timer, predict
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
	def __init__(self, n_gmm_components=50, max_iter=30, tol=1e-4, cov_type='diag'):
		self._n_gmm_components = n_gmm_components
		self.max_iter = max_iter
		self.tol = tol
		self.cov_type = cov_type

	def init(self):
		self._clf = mixture.GaussianMixture(n_components=self._n_gmm_components, max_iter=self.max_iter, init_params='kmeans', covariance_type=self.cov_type)

	def compute(self, data):
		self._clf.fit(data)
		return self._clf.means_, self._clf.weights_, self._clf.covariances_, None
	
	def predict(self, data):
		return self._clf.predict(data)

class GMM_GPU(Feature):
	def __init__(self, n_gmm_components=100, max_iter=30, tol=1e-4, cov_type='diag'):
		self._n_gmm_components = n_gmm_components
		self.max_iter = max_iter
		self.tol = tol
		self.cov_type = cov_type

	def init(self):
		#print("Init")
		self._clf = GMM_GPU_Base(self._n_gmm_components, max_iter=self.max_iter, tol=self.tol, cov_type=self.cov_type)

	def compute(self, data):
		self._clf.fit(data)
		
		return self._clf.means_, self._clf.weights_, self._clf.covariances_, self._clf.inv_covs
	
	def predict(self, data):
		return self._clf.predict(data)
	
class GMM_GPU_Base:
	def __init__(self, num_components, max_iter=30, tol=1e-4, cov_type='diag'):
		self.num_components = num_components
		self.max_iter = max_iter
		self.tol = tol
		self.cov_type = cov_type

	def fit(self, X):
		dev_X = cupy.asarray(X.astype(np.float32))

		means, weights, covs = init_gmm_params(X, self.num_components, cov_type=self.cov_type)

		#t1 = time.time()
		dev_means = cupy.asarray(means.astype(np.float32))
		dev_covs = cupy.asarray(covs.astype(np.float32))
		dev_weights = cupy.asarray(weights.astype(np.float32))
		#t2 = time.time()
		
		with timer('GPU GMM TRAIN'):
			dev_inv_covs, dev_means, dev_weights, dev_covs, dev_lls = train_gmm(dev_X, self.max_iter, self.tol, dev_means, dev_covs, dev_weights, cov_type=self.cov_type)
		
		#means, covs, weights, inv_covs = cupy.asnumpy(dev_means), cupy.asnumpy(dev_covs), cupy.asnumpy(dev_weights), cupy.asnumpy(inv_covs)
		#print(dev_means.dtype, dev_covs.dtype, dev_weights.dtype)
		#print(means.shape, covs.shape, weights.shape)

		self.means_ = dev_means
		self.covariances_ = dev_covs
		self.weights_ = dev_weights
		self.lls = dev_lls
		self.inv_covs = dev_inv_covs
		print("\nLog Likelihood Min-Max:\n\n", np.min(dev_lls), np.max(dev_lls))

		return self
	
	def predict(self, X):
		X = cupy.asarray(X.astype(np.float32))
		return predict(X, self.inv_covs, self.means_, self.weights_, cov_type=self.cov_type)

class GMM_CPU(Feature):
	def __init__(self, n_gmm_components=100, max_iter=30, tol=1e-4, cov_type = 'diag'):
		self._n_gmm_components = n_gmm_components
		self.max_iter = max_iter
		self.tol = tol
		self.cov_type = cov_type

	def init(self):
		self._clf = GMM_CPU_Base(self._n_gmm_components, max_iter=self.max_iter, tol=self.tol, cov_type=self.cov_type)

	def compute(self, data):
		self._clf.fit(data)
		return self._clf.means_, self._clf.weights_, self._clf.covariances_

	def predict(self, data):
		return self._clf.predict(data)

class GMM_CPU_Base:
	def __init__(self, num_components, max_iter=30, tol=1e-4, cov_type = 'diag'):
		self.num_components = num_components
		self.max_iter = max_iter
		self.tol = tol
		self.cov_type = cov_type

	def fit(self, X):
		means, weights, covs = init_gmm_params(X, self.num_components, cov_type=self.cov_type)

		dev_X = X.astype(np.float32)
		dev_means = means.astype(np.float32)
		dev_covs = covs.astype(np.float32)
		dev_weights = weights.astype(np.float32)

		with timer('CPU GMM TRAIN'):
			inv_covs, dev_means, dev_weights, dev_covs, dev_lls = train_gmm(dev_X, self.max_iter, self.tol, dev_means, dev_covs, dev_weights, cov_type=self.cov_type)
				
		self.means_ = dev_means
		self.covariances_ = dev_covs
		self.weights_ = dev_weights
		self.lls = dev_lls
		self.inv_covs = inv_covs

		#print("\nLog Likelihood Min-Max:\n\n", np.min(dev_lls), np.max(dev_lls))

		return self
	
	def predict(self, X):
		return predict(X, self.inv_covs, self.means_, self.weights_, cov_type=self.cov_type)

class OneClassSVM(Feature):
    """Feature points extraction using One class SVM
    Args:
        ndim (int): The dimension of samples.
        sigma (float): Veriance of the gaussian distribution made from parameters of SVM.
        gamma (float, optional): Coefficient for RBF kernel.
        nu (float, optional): An upper bound on the fraction of training errors
            and a lower bound of the fraction of support vectors.
        delta (float, optional): Anealing parameter for optimization.
    """
    def __init__(self, ndim, sigma, gamma=0.5, nu=0.05, delta=10.0):
        self._ndim = ndim
        self._sigma = sigma
        self._gamma = gamma
        self._nu = nu
        self._delta = delta

    def init(self):
        self._clf = svm.OneClassSVM(nu=self._nu, kernel="rbf", gamma=self._gamma)

    def compute(self, data):
        self._clf.fit(data)
        z = np.power(2.0 * np.pi * self._sigma**2, self._ndim * 0.5)
        return self._clf.support_vectors_, self._clf.dual_coef_[0] * z

    def annealing(self):
        self._gamma *= self._delta