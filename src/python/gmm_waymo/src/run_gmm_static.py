import numpy as np
import time
import open3d as o3

from gmm import GMM_CPU, GMM_Sklearn, GMM_GPU
from gmm_impl import predict
import cupy
from scipy.stats import multivariate_normal

filepath = '../data/bunny.pcd'
#filepath = '../data/dragon.ply'
#filepath = '../data/lounge.ply'

n_gmm_components = 50
max_iter = 50
cov_type = 'spherical'

#voxel = 2.5

#voxel = 0.22
#voxel = 0.07
#voxel = 0.032
#voxel = 0.0225
#voxel = 0.014
voxel = 0.0015

source = o3.read_point_cloud(filepath)
source = o3.voxel_down_sample(source, voxel_size=voxel)
print(source)
source_np = np.asarray(source.points)


all_colors = np.array([np.random.uniform(0, 1, 3) for _ in range(n_gmm_components)])

gmm = GMM_GPU(n_gmm_components=n_gmm_components, max_iter=50, cov_type='spherical')
#gmm = GMM_CPU(n_gmm_components=n_gmm_components, max_iter=50, cov_type='spherical')
#gmm = GMM_Sklearn(n_gmm_components=n_gmm_components, max_iter=50, cov_type='spherical')

gmm.init()

start = time.time()
gmm.compute(source_np)
end = time.time()

print(f"Num Points:{source_np.shape[0]}, Num GMM Components:{n_gmm_components}, Time Taken:{end - start}")

#gmm_idxs = map_points_to_gmm(source_np, means, weights, covs, inv_covs)
gmm_idxs = gmm.predict(source_np)
gmm_idxs = cupy.asnumpy(gmm_idxs)
source_colors = all_colors[gmm_idxs, :]

vis = o3.Visualizer()
vis.create_window()
opt = vis.get_render_option()
opt.background_color = np.asarray([0.0, 0.0167, 0.1186])
ctr = vis.get_view_control()
ctr.change_field_of_view(step=270)

source.colors = o3.utility.Vector3dVector(source_colors)

vis.add_geometry(source)
vis.run()