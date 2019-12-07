import transformations as trans
import numpy as np

def diff_rot_from_quaternion(q):
	"""Differencial rotation matrix from quaternion.
	dR(q)/dq = [dR(q)/dq0, dR(q)/dq1, dR(q)/dq2, dR(q)/dq3]
	Args:
		q (numpy.ndarray): Quaternion.
	"""
	rot = trans.quaternion_matrix(q)[:3, :3]
	q2 = np.square(q)
	z = np.sum(q2)
	z2 = z * z
	d_rot = np.zeros((4, 3, 3))
	d_rot[0, 0, 0] = 4 * q[0] * (q2[2] + q2[3]) / z2
	d_rot[1, 0, 0] = 4 * q[1] * (q2[2] + q2[3]) / z2
	d_rot[2, 0, 0] = -4 * q[2] * (q2[1] + q2[0]) / z2
	d_rot[3, 0, 0] = -4 * q[3] * (q2[1] + q2[0]) / z2

	d_rot[0, 1, 1] = 4 * q[0] * (q2[1] + q2[3]) / z2
	d_rot[1, 1, 1] = -4 * q[1] * (q2[2] + q2[0]) / z2
	d_rot[2, 1, 1] = 4 * q[2] * (q2[1] + q2[3]) / z2
	d_rot[3, 1, 1] = -4 * q[3] * (q2[2] + q2[0]) / z2

	d_rot[0, 2, 2] = 4 * q[0] * (q2[1] + q2[2]) / z2
	d_rot[1, 2, 2] = -4 * q[1] * (q2[3] + q2[0]) / z2
	d_rot[2, 2, 2] = -4 * q[2] * (q2[1] + q2[2]) / z2
	d_rot[3, 2, 2] = 4 * q[3] * (q2[3] + q2[0]) / z2

	d_rot[0, 0, 1] = -2 * q[3] / z - 2 * q[0] * rot[0, 1] / z2
	d_rot[1, 0, 1] = 2 * q[2] / z - 2 * q[1] * rot[0, 1] / z2
	d_rot[2, 0, 1] = 2 * q[1] / z - 2 * q[2] * rot[0, 1] / z2
	d_rot[3, 0, 1] = -2 * q[0] / z - 2 * q[3] * rot[0, 1] / z2

	d_rot[0, 0, 2] = 2 * q[2] / z - 2 * q[0] * rot[0, 2] / z2
	d_rot[1, 0, 2] = 2 * q[3] / z - 2 * q[1] * rot[0, 2] / z2
	d_rot[2, 0, 2] = 2 * q[0] / z - 2 * q[2] * rot[0, 2] / z2
	d_rot[3, 0, 2] = 2 * q[1] / z - 2 * q[3] * rot[0, 2] / z2

	d_rot[0, 1, 0] = 2 * q[3] / z - 2 * q[0] * rot[1, 0] / z2
	d_rot[1, 1, 0] = 2 * q[2] / z - 2 * q[1] * rot[1, 0] / z2
	d_rot[2, 1, 0] = 2 * q[1] / z - 2 * q[2] * rot[1, 0] / z2
	d_rot[3, 1, 0] = 2 * q[0] / z - 2 * q[3] * rot[1, 0] / z2

	d_rot[0, 1, 2] = -2 * q[1] / z - 2 * q[0] * rot[1, 2] / z2
	d_rot[1, 1, 2] = -2 * q[0] / z - 2 * q[1] * rot[1, 2] / z2
	d_rot[2, 1, 2] = 2 * q[3] / z - 2 * q[2] * rot[1, 2] / z2
	d_rot[3, 1, 2] = 2 * q[2] / z - 2 * q[3] * rot[1, 2] / z2

	d_rot[0, 2, 0] = -2 * q[2] / z - 2 * q[0] * rot[2, 0] / z2
	d_rot[1, 2, 0] = 2 * q[3] / z - 2 * q[1] * rot[2, 0] / z2
	d_rot[2, 2, 0] = -2 * q[0] / z - 2 * q[2] * rot[2, 0] / z2
	d_rot[3, 2, 0] = 2 * q[1] / z - 2 * q[3] * rot[2, 0] / z2

	d_rot[0, 2, 1] = 2 * q[1] / z - 2 * q[0] * rot[2, 1] / z2
	d_rot[1, 2, 1] = 2 * q[0] / z - 2 * q[1] * rot[2, 1] / z2
	d_rot[2, 2, 1] = 2 * q[3] / z - 2 * q[2] * rot[2, 1] / z2
	d_rot[3, 2, 1] = 2 * q[2] / z - 2 * q[3] * rot[2, 1] / z2

	return d_rot