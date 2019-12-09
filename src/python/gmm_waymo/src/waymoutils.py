import tensorflow as tf
import numpy as np
tf.enable_eager_execution()
import open3d as o3
import cupy

class WaymoLIDARVisCallback(object):
	"""Display Stream of LIDAR Points & GMM

	Args:
		save (bool, optional): If this flag is True,
			each iteration image is saved in a sequential number.
		keep_window (bool, optional): If this flag is True,
			the drawing window blocks after registration is finished.
	"""
	def __init__(self, save=False,
				 keep_window=True):
		#Create Visualizer Window etc
		self._vis = o3.Visualizer()
		self._vis.create_window()
		opt = self._vis.get_render_option()
		opt.background_color = np.asarray([0.0, 0.0167, 0.1186])
		opt.point_size = 1.5
		self._save = save
		self._keep_window = keep_window
		self._cnt = 0
		self._currpc = o3.geometry.PointCloud()

	def __del__(self):
		if self._keep_window:
			self._vis.run()
		self._vis.destroy_window()

	def np_to_pc(self, pts, colors=None):
		self._currpc.points = o3.utility.Vector3dVector(pts)
		if colors is not None:
			self._currpc.colors = o3.utility.Vector3dVector(colors)
		
	def __call__(self, newpoints, colors=None, addpc=False):
		if addpc:
			self._vis.add_geometry(newpoints)
		else:
			#Convert Points into pointcloud 
			self.np_to_pc(newpoints, colors)
			if(self._cnt == 0):
				self._vis.add_geometry(self._currpc)
		self._vis.update_geometry()
		self._vis.poll_events()
		self._vis.update_renderer()
		if self._save:
			self._vis.capture_screen_image("image_%04d.jpg" % self._cnt)
		self._cnt += 1
	
def convert_np_to_pc(np_pts):
	pc = o3.geometry.PointCloud() 
	pc.points = o3.utility.Vector3dVector(np_pts)
	return pc

class WaymoLIDARPair(object):
	"""Get a pair of LIDAR frames (prev, curr)"""
	def __init__(self, skip=0, max_frames = 150, voxel_size = 0.5, gpu = True, filename='../waymodata/segment-10206293520369375008_2796_800_2816_800_with_camera_labels.tfrecord'):
		self._ptr = 1
		self.points_list = np.load(filename, allow_pickle=True)
		self.voxel_size = voxel_size

		self.convert_and_downsample()

		o3.write_point_cloud("test_waymo.pcd", self.pc_list[2])

		self.gpu = gpu
		if gpu:
			self.copy_pc_to_gpu()

	def next_pair(self):
		if(self._ptr < len(self.points_list)):
			if not self.gpu:
				ret_np = (self.points_list[self._ptr-1], self.points_list[self._ptr], self.points_list[self._ptr-1], self.points_list[self._ptr])
			else:
				ret_np = (self.points_list_gpu[self._ptr-1], self.points_list_gpu[self._ptr], self.points_list[self._ptr - 1], self.points_list[self._ptr])

			self._ptr+=1

			return ret_np[0], ret_np[1], ret_np[2], ret_np[3], False
		else:
			return None, None, None, None, True
	
	def copy_pc_to_gpu(self):
		self.points_list_gpu = []
		for i in range(len(self.points_list)):
			pc_np = self.points_list[i]
			self.points_list_gpu.append(cupy.asarray(pc_np.astype(np.float32)))

	def convert_and_downsample(self):
		self.pc_list = []
		self.points_list_ds = []
		for pc_np in self.points_list:
			pc = convert_np_to_pc(pc_np)
			#print(pc)
			pc = o3.voxel_down_sample(pc, voxel_size=self.voxel_size)
			#print(pc)
			self.pc_list.append(pc)	
			self.points_list_ds.append(np.asarray(pc.points))

		self.points_list = self.points_list_ds

class WaymoLIDARPairReg(object):
	"""Get a pair of LIDAR frames (prev, curr)"""
	def __init__(self, skip=0, max_frames = 150, gpu = False, filename='../waymodata/segment-10206293520369375008_2796_800_2816_800_with_camera_labels.tfrecord'):
		self._ptr = 1
		self.points_list = np.load(filename, allow_pickle=True)
		self.gpu = gpu
		if gpu:
			self.copy_pc_to_gpu()

		self.pc_list = []
		for pc_np in self.points_list:
			pc = convert_np_to_pc(pc_np)
			self.pc_list.append(pc)
				
	def next_pair(self):
		if(self._ptr < len(self.points_list)):
			ret_np = (self.pc_list[self._ptr-1], self.pc_list[self._ptr])

			self._ptr+=1

			return ret_np[0], ret_np[1], False
		else:
			return None, None, True
	
	def copy_pc_to_gpu(self):
		self.points_list_gpu = []
		for i in range(self.points_list.shape[0]):
			pc_np = self.points_list[i]
			self.points_list_gpu.append(cupy.asarray(pc_np.astype(np.float32)))
	
