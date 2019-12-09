from __future__ import print_function
from __future__ import division

from sklearn import cluster
from scipy.optimize import minimize
import open3d as o3
import numpy as np
import gmm as ft
import cost_functions as cf
import time
import transformations as trans


class L2DistRegistration(object):
	"""L2 distance registration class
	This algorithm expresses point clouds as mixture gaussian distributions and
	performs registration by minimizing the distance between two distributions.
	Args:
		source (numpy.ndarray): Source point cloud data.
		feature_gen (probreg.features.Feature): Generator of mixture gaussian distribution.
		cost_fn (probreg.cost_functions.CostFunction): Cost function to caliculate L2 distance.
		sigma (float, optional): Scaling parameter for L2 distance.
		delta (float, optional): Annealing parameter for optimization.
		use_estimated_sigma (float, optional): If this flag is True,
			sigma estimates from the source point cloud.
	"""
	def __init__(self, source, feature_gen, cost_fn,
				 sigma=1.0, delta=0.9,
				 use_estimated_sigma=True):
		self._source = source
		self._feature_gen = feature_gen
		self._cost_fn = cost_fn
		self._sigma = sigma
		self._delta = delta
		self._use_estimated_sigma = use_estimated_sigma
		self._callbacks = []
		if not self._source is None and self._use_estimated_sigma:
			self._estimate_sigma(self._source)

	def set_source(self, source):
		self._source = source
		if self._use_estimated_sigma:
			self._estimate_sigma(self._source)

	def set_callbacks(self, callbacks):
		self._callbacks.extend(callbacks)

	def _estimate_sigma(self, data):
		ndata, ndim = data.shape
		data_hat = data - np.mean(data, axis=0)
		self._sigma = np.power(np.linalg.det(np.dot(data_hat.T, data_hat) / (ndata - 1)), 1.0 / (2.0 * ndim))
		print("Estimated Sigma: ", self._sigma)

	def _annealing(self):
		self._sigma *= self._delta

	def optimization_cb(self, x):
		tf_result = self._cost_fn.to_transformation(x)
		for c in self._callbacks:
			c(tf_result)

	def registration(self, target, maxiter=1, tol=1.0e-3,
					 opt_maxiter=10, opt_tol=1.0e-5):

		start = time.time()
		f = None
		x_ini = self._cost_fn.initial()
		self._feature_gen.init()
		
		start_gmm_1 = time.time()
		mu_target, phi_target = self._feature_gen.compute(target)
		end_gmm_1 = time.time()

		mu_target = mu_target
		phi_target = phi_target * 1e3

		# pcd.points = o3.utility.Vector3dVector(mu_target)
		# pcd.paint_uniform_color([0,1,0])

		#print(phi_target)

		for _ in range(maxiter):
			start_gmm_2 = time.time()
			mu_source, phi_source = self._feature_gen.compute(self._source)
			end_gmm_2 = time.time()

			mu_source = mu_source
			phi_source = phi_source * 1e3

			#print(phi_source)

			# pcd2.points = o3.utility.Vector3dVector(mu_source)
			# pcd2.paint_uniform_color([1,0,0])
			
			#print(mu_source.shape)
			#print("Initial: ", x_ini)

			args = (mu_source, phi_source,
					mu_target, phi_target, self._sigma)
			t1 = time.time()
			res = minimize(self._cost_fn,
						   x_ini,
						   args=args,
						   method='BFGS', jac=True,
						   tol=opt_tol,
						   options={'maxiter': opt_maxiter},
						   callback=self.optimization_cb)
			t2 = time.time()
			print("Optimizer TIME: ", t2-t1)
			self._annealing()
			self._feature_gen.annealing()
			if not f is None and abs(res.fun - f) < tol:
				break
			f = res.fun
			x_ini = res.x

		end = time.time()

		print("Overall GMM Time taken: ", end_gmm_2 + end_gmm_1 - start_gmm_2 - start_gmm_1)
		print("Overall Time taken: ", end - start)
		return self._cost_fn.to_transformation(res.x)

class RigidSVR(L2DistRegistration):
    def __init__(self, source, sigma=1.0, delta=0.9,
                 gamma=0.5, nu=0.1, use_estimated_sigma=True):
        super(RigidSVR, self).__init__(source,
                                       ft.OneClassSVM(source.shape[1],
                                                      sigma, gamma, nu),
                                       cf.RigidCostFunction(),
                                       sigma, delta,
                                       use_estimated_sigma)

    def _estimate_sigma(self, data):
        super(RigidSVR, self)._estimate_sigma(data)
        self._feature_gen._sigma = self._sigma
        self._feature_gen._gamma = 1.0 / (2.0 * np.square(self._sigma))

class RigidGMMReg(L2DistRegistration):
	def __init__(self, source, sigma=1.0, delta=0.9,
				 n_gmm_components=50, use_estimated_sigma=True):
		print(source.shape)
		n_gmm_components = min(n_gmm_components, int(source.shape[0] * 0.8))
		print("Number of components: ", n_gmm_components)
		super(RigidGMMReg, self).__init__(source, ft.GMM_GPU(n_gmm_components, max_iter=10),
										  cf.RigidCostFunction(),
										  sigma, delta,
										  use_estimated_sigma)

def registration_gmmreg(source, target, tf_type_name='rigid',
						callbacks=[], **kargs):
	cv = lambda x: np.asarray(x.points if isinstance(x, o3.geometry.PointCloud) else x)
	if tf_type_name == 'rigid':
		gmmreg = RigidGMMReg(cv(source), **kargs)
	else:
		raise ValueError('Unknown transform type %s' % tf_type_name)
	gmmreg.set_callbacks(callbacks)
	return gmmreg.registration(cv(target))

def registration_svr(source, target, tf_type_name='rigid',
                     maxiter=1, tol=1.0e-3,
                     opt_maxiter=50, opt_tol=1.0e-3,
                     callbacks=[], **kargs):
    cv = lambda x: np.asarray(x.points if isinstance(x, o3.geometry.PointCloud) else x)
    if tf_type_name == 'rigid':
        svr = RigidSVR(cv(source), **kargs)
    else:
        raise ValueError('Unknown transform type %s' % tf_type_name)
    svr.set_callbacks(callbacks)
    return svr.registration(cv(target), maxiter, tol, opt_maxiter, opt_tol)

pcd = o3.geometry.PointCloud()
pcd2 = o3.geometry.PointCloud()

if __name__ == "__main__":
	# from probreg import callbacks
	# import utils
	# import transformations as trans

	# source, target = utils.prepare_source_and_target_rigid_3d('bunny.pcd')

	# #cbs = [callbacks.Open3dVisualizerCallback(source, target)]
	# objective_type = 'pt2pt'
	# tf_param = registration_gmmreg(source, target, callbacks=[])
	# rot = trans.identity_matrix()
	# rot[:3, :3] = tf_param.rot
	# print("result: ", np.rad2deg(trans.euler_from_matrix(rot)),
	# 	tf_param.scale, tf_param.t)

	from probreg import gmmtree
	import copy
	#from probreg import transformation

	source = o3.read_point_cloud('waymo1.pcd')
	#target = o3.read_point_cloud("waymo5.pcd")
	#source = o3.read_point_cloud('bunny.pcd')
	target = copy.deepcopy(source)
	# transform target point cloud
	th = np.deg2rad(30.0)
	target.transform(np.array([[np.cos(th), -np.sin(th), 0.0, 0.0],
							[np.sin(th), np.cos(th), 0.0, 0.0],
							[0.0, 0.0, 1.0, 0.0],
							[0.0, 0.0, 0.0, 1.0]]))
	source = o3.voxel_down_sample(source, voxel_size=0.6)
	target = o3.voxel_down_sample(target, voxel_size=0.6)

	# source = o3.voxel_down_sample(source, voxel_size=0.005)
	# target = o3.voxel_down_sample(target, voxel_size=0.005)

	# compute cpd registration
	#tf_param, _ = gmmtree.registration_gmmtree(source, target)

	#tf_param = registration_svr(source, target, callbacks=[])
	tf_param = registration_gmmreg(source, target, callbacks=[])

	rot = trans.identity_matrix()
	rot[:3, :3] = tf_param.rot
	print("result: ", np.rad2deg(trans.euler_from_matrix(rot)), tf_param.scale, tf_param.t)

	result = copy.deepcopy(source)
	result.points = tf_param.transform(result.points)

	# draw result
	source.paint_uniform_color([1, 0, 0])
	target.paint_uniform_color([0, 1, 0])
	result.paint_uniform_color([0, 0, 1])
	o3.draw_geometries([source, target, result])
	#o3.draw_geometries([pcd, pcd2])