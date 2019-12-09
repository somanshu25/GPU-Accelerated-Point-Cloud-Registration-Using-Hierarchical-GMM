import numpy as np
import time
import open3d as o3

from gmm import GMM_CPU, GMM_Sklearn, GMM_GPU
from gmm_impl import predict
import cupy
from scipy.stats import multivariate_normal

from waymoutils import WaymoLIDARPair, convert_np_to_pc, WaymoLIDARVisCallback

#voxel = 2.5
#voxel = 0.7
#voxel = 0.3
voxel = 0.18
#voxel = 0.09

waymopair = WaymoLIDARPair(max_frames=100, voxel_size=voxel, filename='../data/waymo_pcs_100k.npy', gpu=True)
vis = WaymoLIDARVisCallback()

NUM_COMPONENTS = 50
MAX_ITER = 50
COV_TYPE = 'spherical'

all_colors = np.array([np.random.uniform(0, 1, 3) for _ in range(NUM_COMPONENTS)])

fit_gmm_every = 10
num_iters = 0

np.random.seed(5)

while True:
	t1 = time.time()
	source_np, _,source_np_cpu,_, done = waymopair.next_pair()
	if done:
		break

	#2: Runn GMM on Source
	if num_iters % fit_gmm_every == 0:

		gmm = GMM_GPU(n_gmm_components=NUM_COMPONENTS, max_iter=MAX_ITER, cov_type=COV_TYPE)
		gmm.init()

		start = time.time()
		gmm.compute(source_np)
		end = time.time()

		print(f"Num Points:{source_np.shape[0]}, Num GMM Components:{NUM_COMPONENTS}, Time Taken:{end - start}")
	
	gmm_idxs = gmm.predict(source_np)

	#3: Visualize
	gmm_idxs = cupy.asnumpy(gmm_idxs)

	source_colors = all_colors[gmm_idxs, :]
	vis(source_np_cpu, source_colors)

	t2 = time.time()
	print("Time per frame: ", t2-t1)

	num_iters += 1