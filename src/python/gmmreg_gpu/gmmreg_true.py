import copy
import numpy as np
import open3d as o3
from sklearn import cluster
from probreg import l2dist_regs, gmmtree, cpd, filterreg
import transformations as trans

# load source and target point cloud
source = o3.read_point_cloud('waymo1.pcd')
source = o3.voxel_down_sample(source, voxel_size=1.5)
print(source)
target = copy.deepcopy(source)
# transform target point cloud
th = np.deg2rad(10.0)
target.transform(np.array([[np.cos(th), -np.sin(th), 0.0, 0.0],
                           [np.sin(th), np.cos(th), 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]]))
#source = o3.voxel_down_sample(source, voxel_size=0.05)
#print(source)
#target = o3.voxel_down_sample(target, voxel_size=0.05)

# compute cpd registration
#tf_param, _, _ = cpd.registration_cpd(source, target)
objective_type = 'pt2pt'
#tf_param, _, _ = filterreg.registration_filterreg(source, target,objective_type=objective_type,sigma2=None)
tf_param = l2dist_regs.registration_svr(source, target)
#tf_param, _ = gmmtree.registration_gmmtree(source, target)
#tf_param = l2dist_regs.registration_gmmreg(source, target, n_gmm_components=50, use_estimated_sigma=False)

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