import sys
sys.path.insert(1, '/home/sawsn/Shiernee/FHN/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/FileIO/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/Utils/src/utils')

import numpy as np
from FileIO import FileIO
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Utils import Utils
import scipy
import argparse


class FHNDomain:
    def __init__(self, coord=None, sph_coord=None):
        self.coord = coord
        self.sph_coord = sph_coord

        if self.sph_coord is None:
            self.no_pt = len(self.coord)
        elif self.coord is None:
            self.no_pt = len(self.sph_coord)
        elif self.sph_coord is not None or self.coord is not None:
            self.no_pt = len(self.coord)

        self.V = np.zeros([self.no_pt, 1])
        self.v = np.zeros([self.no_pt, 1])
        self.t = np.zeros([self.no_pt, 1])
        self.D = np.ones([self.no_pt, 1])
        self.c = np.ones([self.no_pt, 1])

        self.a = np.ones([self.no_pt, 1])
        self.epsilon = np.ones([self.no_pt, 1])
        self.beta = np.ones([self.no_pt, 1])
        self.gamma = np.ones([self.no_pt, 1])
        self.delta = np.ones([self.no_pt, 1])
        self.applied_current = np.ones([self.no_pt, 1])

        self.local_axis1 = None
        self.local_axis2 = None

        self.region_to_apply_current = None

        return

    def assign_local_axis1(self, local_axis1):
        self.local_axis1 = local_axis1
        return

    def assign_local_axis2(self, local_axis2):
        self.local_axis2 = local_axis2
        return

    def set_initial_V_sph(self, sph_coord, gaussian_2d):
        """

        :param coord: 2D array
        :param gaussian_2d: 1D array
        :return: void
        """
        if len(sph_coord) != len(gaussian_2d):
            raise ValueError("coord must have same length as gaussian_2d")

        phi_min, phi_max, theta_min, theta_max = sph_coord[:, 1].min(), sph_coord[:, 1].max(), \
                                                 sph_coord[:, 2].min(), sph_coord[:, 2].max()
        idx_within_region = np.argwhere((self.sph_coord[:, 1] >= phi_min) & (self.sph_coord[:, 1] <= phi_max) \
                                        & (self.sph_coord[:, 2] >= theta_min) & (self.sph_coord[:, 2] <= theta_max))
        idx_within_region = idx_within_region.squeeze()

        f = scipy.interpolate.Rbf(sph_coord[:, 0], sph_coord[:, 1], sph_coord[:, 2], gaussian_2d)
        self.V[idx_within_region] = f(self.sph_coord[idx_within_region, 0], self.sph_coord[idx_within_region, 1],
                                      self.sph_coord[idx_within_region, 2]).reshape([-1, 1])
        return

    def set_initial_v_sph(self, sph_coord, gaussian_2d):
        """

        :param coord: 2D array
        :param gaussian_2d: 1D array
        :return: void
        """
        if len(sph_coord) != len(gaussian_2d):
            raise ValueError("coord must have same length as gaussian_2d")

        phi_min, phi_max, theta_min, theta_max = sph_coord[:, 1].min(), sph_coord[:, 1].max(), \
                                                 sph_coord[:, 2].min(), sph_coord[:, 2].max()
        idx_within_region = np.argwhere((self.sph_coord[:, 1] >= phi_min) & (self.sph_coord[:, 1] <= phi_max) \
                                        & (self.sph_coord[:, 2] >= theta_min) & (self.sph_coord[:, 2] <= theta_max))
        idx_within_region = idx_within_region.squeeze()

        f = scipy.interpolate.Rbf(sph_coord[:, 0], sph_coord[:, 1], sph_coord[:, 2], gaussian_2d)
        self.v[idx_within_region] = f(self.sph_coord[idx_within_region, 0], self.sph_coord[idx_within_region, 1],
                                      self.sph_coord[idx_within_region, 2]).reshape([-1, 1])
        return

    def set_initial_V(self, coord, gaussian_2d):
        """

        :param coord: 2D array
        :param gaussian_2d: 1D array
        :return: void
        """
        if len(coord) != len(gaussian_2d):
            raise ValueError ("coord must have same length as gaussian_2d")

        xmin, xmax, ymin, ymax = coord[:, 0].min(), coord[:, 0].max(), \
                                 coord[:, 1].min(), coord[:, 1].max()
        idx_within_region = np.argwhere((self.coord[:, 0] >= xmin) & (self.coord[:, 0] <= xmax) \
                                        & (self.coord[:, 1] >= ymin) & (self.coord[:, 1] <= ymax))
        idx_within_region = idx_within_region.squeeze()

        coord, idx = np.unique(coord, axis=0, return_index=True)
        gaussian_2d = gaussian_2d[idx]

        f = scipy.interpolate.Rbf(coord[:, 0], coord[:, 1], coord[:, 2], gaussian_2d[:])
        self.V[idx_within_region] = f(self.coord[idx_within_region, 0], self.coord[idx_within_region, 1],
                                      self.coord[idx_within_region, 2]).reshape([-1, 1])
        return

    def set_initial_v(self, coord, gaussian_2d):
        """

        :param coord: 2D array
        :param gaussian_2d:  1D array
        :return: void
        """
        if len(coord) != len(gaussian_2d):
            raise ValueError("coord must have same length as gaussian_2d")

        xmin, xmax, ymin, ymax = coord[:, 0].min(), coord[:, 0].max(), \
                                 coord[:, 1].min(), coord[:, 1].max()

        idx_within_region = np.argwhere((self.coord[:, 0] >= xmin) & (self.coord[:, 0] <= xmax) \
                                        & (self.coord[:, 1] >= ymin) & (self.coord[:, 1] <= ymax))
        idx_within_region = idx_within_region.squeeze()

        f = scipy.interpolate.Rbf(coord[:, 0], coord[:, 1], coord[:, 2], gaussian_2d)
        self.v[idx_within_region] = f(self.coord[idx_within_region, 0], self.coord[idx_within_region, 1],
                                      self.coord[idx_within_region, 2]).reshape([-1, 1])

        return

    def set_D(self, D_array):
        """

        :param D_array:
        :return:
        """
        return

    def set_c(self, c_array):
        """

        :param c_array:
        :return:
        """
        return

    def set_a(self, a_array):
        """

        :param a_array: a_array
        :return:
        """
        return

    def set_epsilon(self, epsilon_array):
        '''

        :param epsilon_array: epsilon_array
        :return:
        '''
        return

    def set_beta(self, beta_array):
        """

        :param beta_array:
        :return:
        """
        return

    def set_gamma(self, gamma_array):
        """

        :param gamma_array:
        :return:
        """
        return

    def set_delta(self, delta_array):
        """

        :param delta_array:
        :return:
        """
        return

    def applied_current(self, applied_current):
        """

        :param applied_current:
        :return:
        """
        return

    def set_region_to_apply_current(self, region_to_apply_current):
        self.region_to_apply_current = region_to_apply_current
        return

    def instance_to_dict(self):
        instances = \
            {'coord': self.coord,
             'region_to_apply_current': self.region_to_apply_current,
             't': self.t,
             'V': self.V,
             'v': self.v,
             'D': self.D,
             'c': self.c,
             'a': self.a,
             'epsilon': self.epsilon,
             'beta': self.beta,
             'gamma': self.gamma,
             'delta': self.delta,
             'applied_current': self.applied_current,
             'local_axis1': self.local_axis1,
             'local_axis2': self.local_axis2
             }

        return instances


class BoundaryCondition:
    def __init__(self):
        return

    def inverse_gaussian_2D(self, xmin, xmax, ymin, ymax,
                            x0=0, y0=0, sigma_x=0.1, sigma_y=0.1, a=1):
        """

        :param xmin:
        :param xmax:
        :param ymin:
        :param ymax:
        :param x0:
        :param y0:
        :param sigma_x:
        :param sigma_y:
        :param a:
        :return:
        """
        x, y = np.meshgrid(np.linspace(xmin, xmax,  100), np.linspace(ymin, ymax, 100))
        coord = np.zeros([len(x.flatten()), 3])
        coord[:, 0], coord[:, 1] = x.flatten(), y.flatten()
        f = -1 * a * np.exp(-1 * (((x - x0) ** 2 / 2 / sigma_x ** 2) + ((y - y0) ** 2 / 2 / sigma_y ** 2)))
        f = f.flatten()

        return coord, f

    def gaussian_2D(self, xmin, xmax, ymin, ymax,
                    x0=0, y0=0, sigma_x=0.1, sigma_y=0.1, a=1):
        """

        :param left: float
        :param right: float
        :param bottom: float
        :param top: float
        :param x0: float
        :param y0: float
        :param sigma_x: float
        :param sigma_y: float
        :param a: amplitude - float
        :return: array, array
        """
        x, y = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
        coord = np.zeros([len(x.flatten()), 3])
        coord[:, 0], coord[:, 1] = x.flatten(), y.flatten()
        f = a * np.exp(-1 * (((x - x0)**2 / 2 / sigma_x **2) + ((y - y0)**2 / 2 / sigma_y **2)))
        f = f.flatten()

        return coord, f

    def normalize(self, gaussian):
        g = (gaussian - gaussian.min())
        g = g/g.max()
        return g


class GeneratePoints:
    def __init__(self):
        return

    def generate_Niederreiter_datasets(self, no_pt, max_x, max_y):
        """

        :param filename: Niederreiter.csv
        :return: 2D array
        """
        filename = '../data/case2_NIEDERREITER2_DATASET/niederreiter2_02_10000.csv'
        dataframe_raw = pd.read_csv(filename, header=None)
        dataframe_raw = dataframe_raw.drop([0], axis=1)
        dataframe_raw.columns = ['x', 'y']

        # scaling
        dataframe_raw['x'] = dataframe_raw['x'] * max_x
        dataframe_raw['y'] = dataframe_raw['y'] * max_y

        dataframe = dataframe_raw.head(no_pt)
        coord = np.zeros([no_pt, 3])
        coord[:, 0], coord[:, 1] = dataframe['x'].values, dataframe['y'].values

        return coord

    def generate_point_regular_grid(self, no_pt, max_x, max_y):
        """

        :param dx: flaot
        :param max_x: flaot
        :param max_y: flaot
        :return: 2D array
        """
        x = np.linspace(0, max_x, np.sqrt(no_pt))
        y = np.linspace(0, max_y, np.sqrt(no_pt))
        X, Y = np.meshgrid(x, y)
        coord = np.zeros([len(X.flatten()), 3])
        coord[:, 0] = X.flatten()
        coord[:, 1] = Y.flatten()

        return coord

    def generate_points_2D(self, no_pt, max_x, max_y):
        """

        :param no_pt: int
        :param max_x: float
        :param max_y: float
        :return: 2D array
        """
        # no_pt = int(max_x / dx + 1) * int(max_x / dx + 1)
        x = np.random.rand(no_pt, ) * max_x
        y = np.random.rand(no_pt, ) * max_y
        coord = np.zeros([no_pt, 3])
        coord[:, 0] = x
        coord[:, 1] = y

        return coord

    def generate_points_sphere(self, no_pt, sph_radius):
        """

        :param n_point: int
        :param sph_radius: float
        :return: 2D array
        """
        ut = Utils()
        phi = np.random.uniform(-np.pi, np.pi, no_pt)
        theta = np.random.uniform(0, np.pi, no_pt)
        radius = np.ones([no_pt, ]) * sph_radius

        sph_coord = np.zeros([no_pt, 3])
        sph_coord[:, 0], sph_coord[:, 1], sph_coord[:, 2] = radius, phi, theta
        x, y, z = ut.sph2xyz(sph_coord)

        cart_coord = np.zeros([no_pt, 3])
        cart_coord[:, 0], cart_coord[:, 1], cart_coord[:, 2] = x, y, z

        return cart_coord, sph_coord


class Plot:
    def __init__(self):
        return

    @staticmethod
    def view_2D_V(coord, V, fileio, save_fig=True):
        """

        :param coord: array
        :param V: array
        :param fileio: object
        :param save_fig: boolean
        :return: void
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')

        cbar = ax.scatter(coord[:, 0], coord[:, 1], c=np.squeeze(V), s=10)
        plt.grid()
        fig.colorbar(cbar)
        # plt.show()
        if save_fig is True:
            fileio.save_png_file(i=1, model='FHN', fig_name='Initial_V')

        return

    @staticmethod
    def view_3D_V(coord, V, fileio, save_fig=True):
        x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cbar = ax.scatter(x, y, z, s=20, c=np.squeeze(V))
        fig.colorbar(cbar)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if save_fig is True:
            fileio.save_png_file(i=1, model='FHN', fig_name='Initial_V')
        plt.close()
        return

    @staticmethod
    def view_phi_theta_V(coord, V, fileio, save_fig=True):
        ut = Utils()
        _, phi, theta = ut.xyz2sph(coord)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect(0.5)

        cbar = ax.scatter(theta, phi, c=np.squeeze(V), s=5)
        fig.colorbar(cbar)
        ax.set_xlabel('theta')
        ax.set_ylabel('phi')
        if save_fig is True:
            fileio.save_png_file(i=1, model='FHN', fig_name='Initial_V_phi_theta')
        plt.close()
        return

''' 
NOT USED HERE 
class Utils:
    def __init__(self):
        return

    @staticmethod
    def find_coord_index(global_coord, list_coord_to_be_found):
        """
        :param global_coord: array (n_coord x 3)
        :param list_coord_to_be_found: array of coord to be found
        :return:
        """
        list_coord_to_be_found = np.array(list_coord_to_be_found)
        list_coord_to_be_found = np.reshape(list_coord_to_be_found, [-1, 3])
        indices_list = []
        for tmp_coord in list_coord_to_be_found:
            indices = np.where((tmp_coord == global_coord).all(axis=1))
            indices_list.append(indices[0])
        return indices_list

    def assign_current_to_coord(self, V, coord, region_to_apply_current, applied_current_value):
        coord = coord.copy()
        region_to_apply_current_indices = self.find_coord_index(coord, region_to_apply_current)

        region_to_apply_current_indices = np.squeeze(np.array(region_to_apply_current_indices))
        V[region_to_apply_current_indices] = applied_current_value
        V = np.squeeze(V)
        assert np.ndim(V) == 1, 'applied_current dimension should be 1'
        applied_current = V.copy()

        return applied_current
    
    @staticmethod
    def get_nearets_coord_ind(global_coord, coord_to_find):
        coord_to_find = np.array(coord_to_find)
        coord_to_find = np.reshape(coord_to_find, [1, 3])
        eu_dist = np.linalg.norm(global_coord - coord_to_find, axis=1)
        coord_ind = np.where(eu_dist == np.min(eu_dist))
        return coord_ind
'''


class Sampling:
    def __init__(self, dataframe_raw):
        self.dataframe_raw = dataframe_raw
        self.dataframe = None
        return

    def take_first_n_pts_from_dataframe(self, dataframe_raw, no_pt_to_take):
        """

        :param dataframe_raw: dataframe
        :param no_pt_to_take: integer
        :return:
        """
        self.dataframe = dataframe_raw.head(no_pt_to_take)
        return self.dataframe

    def replace_coord_within_range(self, dataframe, range, new_coord_to_replace):
        """
        :param coord: dataframe
        :param range: list
        :param new_coord_to_replace: list
        :return:
        """
        # === this is to make sure no points is in the region of 48-52
        coord_tmp = dataframe.loc[(dataframe['x'] > range[0]) & (dataframe['x'] < range[1]) & \
                                  (dataframe['y'] > range[2]) & (dataframe['y'] < range[3])]
        idx = coord_tmp.index.values
        dataframe_size = self.dataframe_raw.shape[0] - dataframe.shape[0]

        while idx.shape[0] > 1:
            coord_tmp = dataframe.loc[(dataframe['x'] > range[0]) & (dataframe['x'] < range[1]) & \
                                      (dataframe['y'] > range[2]) & (dataframe['y'] < range[3])]
            idx = coord_tmp.index.values
            df_to_replace = self.dataframe_raw.tail(dataframe_size).sample(n=len(idx))
            dataframe.loc[idx] = df_to_replace.values

        no_pt_to_take = dataframe.shape[0]
        # === replace 4 points with user-defined so that it match with grid simulation initial boundary condition
        rdm_loc = np.random.randint(0, no_pt_to_take, np.shape(new_coord_to_replace)[0])
        dataframe.loc[rdm_loc] = new_coord_to_replace

        assert dataframe.shape == dataframe.drop_duplicates().shape, 'updated dataframe shape not matched with original ' \
                                                                     'dataframe'
        return dataframe


def main_2D_grid(forward_folder, no_pt=81**2, max_x=100, max_y=100, type='grid'):
    """
    - create FHN instances and save as .sav file
    - initial boundary condition generated using gaussian at center with sigma=0.1*max_x and 0.1*max_y, with
    amplitude = 1

    :param forward_folder: str
    :param no_pt: int
    :param max_x: float
    :param max_y: float
    :param type: str
    :return: void
    """
    # ========== FHN Domain instance in 2D ==================
    # === assign forward_folder to save the instances generated ===
    fileio = FileIO()
    fileio.assign_forward_folder(forward_folder)

    # ===== generate points in 2D =======
    gen_pts = GeneratePoints()
    # choose one domain to generate points

    if type == 'Niederreiter':
        coord = gen_pts.generate_Niederreiter_datasets(no_pt=no_pt, max_x=max_x, max_y=max_y)
    if type == 'grid':
        coord = gen_pts.generate_point_regular_grid(no_pt=no_pt, max_x=max_x, max_y=max_y)
    if type == 'scatter':
        coord = gen_pts.generate_points_2D(no_pt=no_pt, max_x=max_x, max_y=max_y)

    localaxis1 = [1, 0, 0]
    localaxis2 = [0, 1, 0]
    local_axis1 = np.tile(localaxis1, (len(coord), 1))
    local_axis2 = np.tile(localaxis2, (len(coord), 1))

    domain = FHNDomain(coord)
    domain.assign_local_axis1(local_axis1=local_axis1)
    domain.assign_local_axis2(local_axis2=local_axis2)

    # === initial condition =====
    bc = BoundaryCondition()
    # xmin, xmax, ymin, ymax = 0.48*max_x, 0.52*max_x, 0.48*max_y, 0.52*max_y
    x0, y0, sigma_x, sigma_y, amplitude = max_x/2, max_y/2, 0.02*max_x, 0.02*max_x, 1
    region_to_apply_current, gaussian_2d = bc.gaussian_2D(0, max_x, 0, max_y,
                                                          x0, y0, sigma_x, sigma_y, amplitude)
    gaussian_2d = bc.normalize(gaussian_2d)

    domain.set_initial_V(region_to_apply_current, gaussian_2d)
    domain.set_region_to_apply_current(region_to_apply_current)

    plot = Plot()
    plot.view_2D_V(domain.coord, domain.V, fileio, save_fig=True)

    # == save instances to .sav file ===
    instance = domain.instance_to_dict()
    fileio.write_generated_instance(instance)
    return


def main_sphere(forward_folder, no_pt=5000, sph_radius=40.0):
    ###############################################################################################
    # ==========  FHN Domain instance in sphere ==================
    # === assign forward_folder to save the instances generated ===
    fileio = FileIO()
    fileio.assign_forward_folder(forward_folder)

    # ===== generate points on sphere ================
    gen_pts = GeneratePoints()
    sph_radius = sph_radius
    coord, sph_coord = gen_pts.generate_points_sphere(no_pt=no_pt, sph_radius=sph_radius)

    local_axis1 = None
    local_axis2 = None

    domain = FHNDomain(coord=coord, sph_coord=sph_coord)
    domain.assign_local_axis1(local_axis1=local_axis1)
    domain.assign_local_axis2(local_axis2=local_axis2)

    # === initial condition =====
    bc = BoundaryCondition()
    phi_min, phi_max, theta_min, theta_max = -np.pi, np.pi, 0, np.pi
    phi_mean, theta_mean, sigma_phi, sigma_theta, amplitude = 0, np.pi / 2, 2 * np.pi, np.pi / 4, 1
    phi_theta_sph_coord, gaussian_2d = bc.inverse_gaussian_2D(phi_min, phi_max, theta_min, theta_max,
                                                              phi_mean, theta_mean, sigma_phi, sigma_theta,
                                                              amplitude)
    gaussian_2d = bc.normalize(gaussian_2d)

    # == rearrange indices ===
    region_to_apply_current_sph = np.zeros_like(phi_theta_sph_coord)
    region_to_apply_current_sph[:, 0] = sph_radius
    region_to_apply_current_sph[:, 1] = phi_theta_sph_coord[:, 0]
    region_to_apply_current_sph[:, 2] = phi_theta_sph_coord[:, 1]

    domain.set_initial_V_sph(region_to_apply_current_sph, gaussian_2d)

    plot = Plot()
    plot.view_phi_theta_V(domain.coord, domain.V, fileio, save_fig=True)
    plot.view_3D_V(domain.coord, domain.V, fileio, save_fig=True)

    # == save instances to .sav file ===
    instance = domain.instance_to_dict()
    fileio.write_generated_instance(instance)
    return


if __name__ == '__main__':

    # forward_folder = '../data/case2_2Dgrid_100/forward_10201pts/'
    forward_folder = '../data/case3_sphere/forward_5000pts/'
    # main_2D_grid(forward_folder, no_pt=101**2, max_x=55, max_y=55, type='grid')
    main_sphere(forward_folder, no_pt=5000, sph_radius=1.0)



    """
    # === READ PREDICTED PARAMETERS FROM SAV  =====
    forward_folder = '../data/case3_sphere/forward/'
    inverse_folder = '../data/case3_sphere/inverse/'

    fileio = FileIO()
    fileio.assign_forward_folder(forward_folder)
    fileio.assign_inverse_folder(inverse_folder)
    i = 1

    fhn_model_instances = fileio.read_physics_model_instance(i, model='fhn')
    fhn_dl_model_instances = fileio.read_inverse_physics_model_instance(i, model='fhn')
    diffusion_model_instances = fileio.read_physics_model_instance(i, model='diffusion')
    diffusion_dl_model_instances = fileio.read_inverse_physics_model_instance(i, model='diffusion')
    point_cloud_instances = fileio.read_point_cloud_instance(i)

    coord = point_cloud_instances['coord']
    t = np.zeros([len(coord), 1])
    V = fhn_model_instances['V0']
    v = fhn_model_instances['v0']
    region_to_apply_current = coord[np.where(V[0] == 1.0)]

    a = fhn_dl_model_instances['tf_weight_a'].numpy().squeeze()
    epsilon = fhn_dl_model_instances['tf_weight_epsilon'].numpy().squeeze()
    beta = fhn_dl_model_instances['tf_weight_beta'].numpy().squeeze()
    gamma = fhn_dl_model_instances['tf_weight_gamma'].numpy().squeeze()
    delta = fhn_dl_model_instances['tf_weight_delta'].numpy().squeeze()
    applied_current = fhn_dl_model_instances['tf_weight_current'].numpy().squeeze()

    D = diffusion_dl_model_instances['tf_weight_D'].numpy().squeeze()
    c = diffusion_dl_model_instances['tf_weight_c'].numpy().squeeze()

    local_axis1 = None
    local_axis2 = None
    
    view_phi_theta_V(coord, V)
    # view_3D_V(coord, V)

    # parse argument NOT USED 
    parser = argparse.ArgumentParser(description='This scripts is to generate FHN instances and '
                                                 'save as .sav file.')
    parser.add_argument('Domain', type=str,
                        help='-2D_Niederreiter, -2D_grid, -2D_scatter, -sphere')
    args = parser.parse_args()
    domain = vars(args).get('Domain')
    print('Generating FHN instance in a domain of {}'.format(domain))
    """


