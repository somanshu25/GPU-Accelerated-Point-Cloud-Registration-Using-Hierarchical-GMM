import abc
import six
import numpy as np
import open3d as o3

@six.add_metaclass(abc.ABCMeta)
class Transformation():
	def __init__(self):
		pass

	def transform(self, points, array_type=o3.utility.Vector3dVector):
		if isinstance(points, array_type):
			return array_type(self._transform(np.asarray(points)))
		return self._transform(points)

	@abc.abstractmethod
	def _transform(self, points):
		return points


class RigidTransformation(Transformation):
	"""Rigid Transformation
	Args:
		rot (numpy.ndarray, optional): Rotation matrix.
		t (numpy.ndarray, optional): Translation vector.
		scale (Float, optional): Scale factor.
	"""
	def __init__(self, rot=np.identity(3),
				 t=np.zeros(3), scale=1.0):
		super(RigidTransformation, self).__init__()
		self.rot = rot
		self.t = t
		self.scale = scale

	def _transform(self, points):
		return self.scale * np.dot(points, self.rot.T) + self.t

	def inverse(self):
		return RigidTransformation(self.rot.T, -np.dot(self.rot.T, self.t),
								   1.0 / self.scale)


def _gauss_transform_direct(source, target, weights, h):
	"""
	\sum_{j} weights[j] * \exp{ - \frac{||target[i] - source[j]||^2}{h^2} }
	"""
	h2 = h * h
	fn = lambda t: np.dot(weights, np.exp(-np.sum(np.square(t - source), axis=1) / h2))
	return np.apply_along_axis(fn, 1, target)

class Direct(object):
	def __init__(self, source, h):
		self._source = source
		self._h = h

	def compute(self, target, weights):
		return _gauss_transform_direct(self._source, target, weights, self._h)


class GaussTransform(object):
	"""Calculate Gauss Transform
	Args:
		source (numpy.ndarray): Source data.
		h (float): Bandwidth parameter of the Gaussian.
		eps (float): Small floating point used in Gauss Transform.
		sw_h (float): Value of the bandwidth parameter to
			switch between direct method and IFGT.
	"""
	def __init__(self, source, h, eps=1.0e-4, sw_h=0.01):
		self._m = source.shape[0]
		self._impl = Direct(source, h)

	def compute(self, target, weights=None):
		"""Compute gauss transform
		Args:
			target (numpy.ndarray): Target data.
			weights (numpy.ndarray): Weights of Gauss Transform.
		"""
		if weights is None:
			weights = np.ones(self._m)
		if weights.ndim == 1:
			return self._impl.compute(target, weights)
		elif weights.ndim == 2:
			return np.r_[[self._impl.compute(target, w) for w in weights]]
		else:
			raise ValueError("weights.ndim must be 1 or 2.")