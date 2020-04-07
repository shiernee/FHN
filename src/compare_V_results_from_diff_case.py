import sys
sys.path.insert(1, '/home/sawsn/Shiernee/FHN/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/PointCloud/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/FileIO/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/Utils/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/Diffusion/src/utils')

import numpy as np
from FileIO import FileIO
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

class Case:
    def __init__(self, forward_folder):
        self.forward_folder = forward_folder

        self.fhn_model_instances= None
        self.diffusion_model_instances = None
        self.point_cloud_instances = None

        self.coord = None
        self.no_pt = None
        self.t = None
        self.V = None
        self.v = None
        self.a = None
        self.epsilon = None
        self.beta = None
        self.delta = None
        self.gamma = None
        self.stimulated_current = None
        self.D = None
        self.c = None

        self.read_instance(self.forward_folder)
        return

    def read_instance(self, forward_folder):
        fileio = FileIO()
        fileio.assign_forward_folder(forward_folder)
        i = 1
        self.fhn_model_instances = fileio.read_physics_model_instance(i, 'fhn')
        self.diffusion_model_instances = fileio.read_physics_model_instance(i, 'diffusion')
        self.point_cloud_instances = fileio.read_point_cloud_instance(i)

        # ========================== get variable  ================================ #
        self.coord = self.point_cloud_instances['coord']
        self.no_pt = self.point_cloud_instances['no_pt']
        self.t = self.fhn_model_instances['t']
        self.V = self.fhn_model_instances['V']
        self.v = self.fhn_model_instances['v']
        self.a = self.fhn_model_instances['a']
        self.delta = self.fhn_model_instances['delta']
        self.gamma = self.fhn_model_instances['gamma']
        self.stimulated_current = np.max(self.fhn_model_instances['applied_current'])
        self.D = self.diffusion_model_instances['D']
        self.c = self.diffusion_model_instances['c']
        return


if __name__ == '__main__':
     scatter_forward_folder = '../data/case2_2Dscatter_100/forward1_fixaxis/'
     scatter = Case(scatter_forward_folder)

     grid_forward_folder = '../data/case2_2Dgrid_100/forward/'
     grid = Case(grid_forward_folder)

     assert scatter.t.shape[0] == grid.t.shape[0], 'grid and scatter total timestep is different'
     total_time_step = scatter.t.shape[0]

     x, y, z = grid.coord[:, 0], grid.coord[:, 1], grid.coord[:, 2]
     max_x, max_y = round(x.max()), round(x.max())
     Xi = np.arange(round(x.min()), round(x.max() + 2), 2)
     Yi = np.arange(round(y.min()), round(y.max() + 2), 2)
     Xi, Yi = np.meshgrid(Xi, Yi)

     for i in range(0, 100, 10):
         fig = plt.figure()
         voltage_to_plot = grid.V[i, :]
         rbf = Rbf(grid.coord[:, 0], grid.coord[:, 1], voltage_to_plot, function='linear', smooth=3)
         V_smooth_grid = rbf(Xi, Yi)

         voltage_to_plot = scatter.V[i, :]
         rbf = Rbf(scatter.coord[:, 0], scatter.coord[:, 1], voltage_to_plot, function='linear', smooth=3)
         V_smooth_scatter = rbf(Xi, Yi)

         ax1 = fig.add_subplot(321)
         ax2 = fig.add_subplot(322)
         ax3 = fig.add_subplot(323)
         ax4 = fig.add_subplot(324)
         ax5 = fig.add_subplot(325)

         ax1.imshow(V_smooth_scatter, extent=[0, max_x, 0, max_y], origin='lower')
         ax2.imshow(V_smooth_grid, extent=[0, max_x, 0, max_y], origin='lower')
         ax3.scatter(scatter.coord[:, 0], scatter.coord[:, 1], c=scatter.V[i, :])
         ax4.scatter(grid.coord[:, 0], grid.coord[:, 1], c=grid.V[i, :])
         ax5.imshow(V_smooth_grid - V_smooth_scatter, origin='lower')

         ax1.set_title('smooth_scatter')
         ax2.set_title('smooth_grid')
         ax3.set_title('scatter')
         ax4.set_title('grid')
         ax5.set_title('error')

         x0, x1 = ax3.get_xlim()
         y0, y1 = ax3.get_ylim()
         ax3.set_aspect(abs(x1 - x0) / abs(y1 - y0))
         ax3.grid()

         x0, x1 = ax4.get_xlim()
         y0, y1 = ax4.get_ylim()
         ax4.set_aspect(abs(x1 - x0) / abs(y1 - y0))


         fig.suptitle('time {}'.format(scatter.t[i]))

         fig.tight_layout()
