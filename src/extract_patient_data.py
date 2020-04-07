import sys
sys.path.insert(1, '/home/sawsn/Shiernee/FHN/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/FileIO/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/Utils/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/PointCloud/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/Database/src/utils')

import numpy as np
import pandas as pd
from FileIO import FileIO
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Utils import Utils
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import pymysql
import pandas as pd
from DatabaseController import DatabaseController


def instance_to_dict(coord, region_to_apply_current, t, V, v, D, c, a, epsilon, beta, gamma, delta,
                     applied_current, local_axis1, local_axis2):
    instance = \
        {'coord': coord,
         'region_to_apply_current': region_to_apply_current,
         't': t,
         'V': V,
         'v': v,
         'D': D,
         'c': c,
         'a': a,
         'epsilon': epsilon,
         'beta': beta,
         'gamma': gamma,
         'delta': delta,
         'applied_current': applied_current,
         'local_axis1': local_axis1,
         'local_axis2': local_axis2
         }

    return instance


def view_3D_V(coord, V):
    x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cbar = ax.scatter(x, y, z, s=100, c=np.squeeze(V), marker='.')
    fig.colorbar(cbar)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    return


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def compute_local_axis(nn_coord, coord):
    nn_coord = nn_coord.copy()
    coord = coord.copy()

    assert np.ndim(nn_coord) == 3, 'nn_coord should be (n_coord, nn_coord, 3 axes)'
    assert np.ndim(coord) == 2, 'nn_coord should be (n_coord, 3 axes)'
    # compute local axis
    v_a = nn_coord[:, 1, :] - coord  # vector A
    v_b = nn_coord[:, 2, :] - coord  # vector B

    v_n = np.cross(v_a, v_b)  ## normal vector to plane AB and interest pt
    v_n = (v_n / np.linalg.norm(v_n))  ##unit vector normal

    local_axis1 = v_a / np.linalg.norm(v_a, axis=1, keepdims=True)
    v_a_d2 = np.cross(v_n, v_a)
    local_axis2 = v_a_d2 / np.linalg.norm(v_a_d2, axis=1, keepdims=True)

    local_axis1, local_axis2 = local_axis1.copy(), local_axis2.copy()

    return local_axis1, local_axis2

if __name__ == '__main__':

    # # == access using database ============
    # # 1. Initiate database connection object
    # db = DatabaseController('admin', '4854197saw', 'sigil_cardiac', '172.20.192.178')
    #
    # # 2a. Retrieve data as pandas DataFrame via DatabaseController.pd_read_sql() function
    # df = db.pd_read_sql('SELECT * FROM case_data;')
    # coord_raw = df[['x', 'y', 'z']].values()
    #

    # === READ PREDICTED PARAMETERS FROM SAV  =====
    forward_folder = '../data/case4_LAF_21_52_36/forward/'
    fileio = FileIO()
    fileio.assign_forward_folder(forward_folder)
    i = 1

    # === read from patients csv ====================
    dataframe = pd.read_csv('{}/coordinates.csv'.format(forward_folder))
    coord_raw = dataframe[['ptx', 'pty', 'ptz']].values

    xtra_dataframe = pd.read_csv('{}/coordinates_to_be_deleted.txt'.format(forward_folder), sep=' ', header=None)
    coord_to_be_deleted = xtra_dataframe.values

    # ====== remove extra coordinates ======
    idx=np.ones(len(coord_raw))
    for value in coord_to_be_deleted:
        aa = np.sum(abs(value - coord_raw), axis=1)
        idx[np.where(aa < 1e-10)[0]] = 0
    idx = np.array(idx).squeeze()
    coord = coord_raw[idx==1]

    no_pt = len(coord)
    t = np.zeros([no_pt, 1])
    V = np.zeros([no_pt, 1])
    v = np.zeros([no_pt, 1])
    D = np.ones([no_pt, 1])
    c = np.ones([no_pt, 1])

    a = np.ones([no_pt, 1])
    epsilon = np.ones([no_pt, 1])
    beta = np.ones([no_pt, 1])
    gamma = np.ones([no_pt, 1])
    delta = np.ones([no_pt, 1])
    applied_current = np.ones([no_pt, 1])

    local_axis1 = None
    local_axis2 = None

    view_3D_V(coord, V)

    ut = Utils()
    r, phi, theta = ut.xyz2sph(coord)
    print('n_pt: {}, ori_dx: {}'.format(len(coord), np.sqrt(np.pi * r.mean()**2 / len(coord))))

    # ============= interpolate points ========
    # n_neighbours = 8
    # nbrs = NearestNeighbors(n_neighbours, algorithm='kd_tree').fit(coord)
    # dist, nn_indices = nbrs.kneighbors(coord)
    # nn_coord = coord[nn_indices]
    # # local_axis1, local_axis2 = compute_local_axis(nn_coord, coord)
    # vec1 = nn_coord[:, int(n_neighbours/2) - 1] - coord
    # vec2 = nn_coord[:, int(n_neighbours/2)] - coord
    #
    # mag1, mag2 = np.linalg.norm(vec1, axis=1), np.linalg.norm(vec2, axis=1)
    # unit_vec1 = vec1 / np.expand_dims(mag1, axis=1)
    # unit_vec2 = vec2 / np.expand_dims(mag2, axis=1)
    #
    # median_pt2pt_dist = np.median(np.array([mag1, mag2]))
    # max_pt2pt_dist = np.array([mag1, mag2]).max()
    # min_pt2pt_dist = np.array([mag1, mag2]).min()
    #
    # thres = 5
    # idx1 = np.argwhere(mag1 > thres).squeeze()
    # idx2 = np.argwhere(mag2 > thres).squeeze()
    #
    # scale = int(median_pt2pt_dist)
    # intp_coord1 = coord[idx1] + unit_vec1[idx1] * scale
    # intp_coord2 = coord[idx1] - unit_vec1[idx1] * scale
    # intp_coord3 = coord[idx2] + unit_vec2[idx2] * scale
    # intp_coord4 = coord[idx2] - unit_vec2[idx2] * scale
    #
    # assert np.ndim(intp_coord1) == 2, 'intp_coord1 must be 2D array'
    # assert np.ndim(intp_coord2) == 2, 'intp_coord1 must be 2D array'
    # assert np.ndim(intp_coord3) == 2, 'intp_coord1 must be 2D array'
    # assert np.ndim(intp_coord4) == 2, 'intp_coord1 must be 2D array'
    #
    # coord_incl_intp_coord = np.array([])
    # coord_incl_intp_coord = np.concatenate((intp_coord1, intp_coord2), axis=0)
    # coord_incl_intp_coord = np.concatenate((coord_incl_intp_coord, intp_coord3), axis=0)
    # coord_incl_intp_coord = np.concatenate((coord_incl_intp_coord, intp_coord4), axis=0)
    # coord_incl_intp_coord = np.concatenate((coord_incl_intp_coord, coord), axis=0)
    # coord = coord_incl_intp_coord.copy()
    #
    # print(coord.shape)
    #
    # no_pt = len(coord)
    # V = np.zeros([no_pt, 1])
    # V = np.zeros([no_pt, 1])
    # v = np.zeros([no_pt, 1])
    # D = np.ones([no_pt, 1])
    # c = np.ones([no_pt, 1])
    #
    # a = np.ones([no_pt, 1])
    # epsilon = np.ones([no_pt, 1])
    # beta = np.ones([no_pt, 1])
    # gamma = np.ones([no_pt, 1])
    # delta = np.ones([no_pt, 1])
    # applied_current = np.ones([no_pt, 1])
    #
    # local_axis1 = None
    # local_axis2 = None
    #
    # view_3D_V(coord_incl_intp_coord, V)

    # ================ assign initial boundary condition ============
    # === point to have slow diffusion / conduction ===========
    idx_to_have_isthmus = random.randrange(0, no_pt)
    nbrs = NearestNeighbors(n_neighbors=8, algorithm='kd_tree').fit(coord)
    dist, nn_indices = nbrs.kneighbors(coord[idx_to_have_isthmus].reshape(1, -1))

    # =================== point to give current ===============
    idx_to_have_activation = random.randrange(0, no_pt)
    V[idx_to_have_activation] = 1.0
    region_to_apply_current = coord[idx_to_have_activation]

    view_3D_V(coord, V)
    # ================================================================

    instance = instance_to_dict(coord, region_to_apply_current, t, V, v, D, c, a, epsilon, beta, gamma,
                                delta, applied_current, local_axis1, local_axis2)

    fileio = FileIO()
    fileio.assign_forward_folder(forward_folder)
    fileio.write_generated_instance(instance)







    '''
    mesh = om.TriMesh()
    vh = []
    for value in coord:
        vh.append(mesh.add_vertex(value))
    fh = mesh.add_face(vh)

    view_phi_theta_V(coord, V)
    view_3D_V(coord_incl_intp_coord, V)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    o3d.visualization.draw_geometries([pcd])
    tetra_mesh, pt_map = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.1, None)


    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    # Radius oulier removal
    cl, ind = pcd.remove_radius_outlier(nb_points=10, radius=5)
    display_inlier_outlier(pcd, ind)


    # Statistical outlier removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    '''
