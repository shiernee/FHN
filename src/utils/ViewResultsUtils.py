import sys
sys.path.insert(1, '/home/sawsn/Shiernee/Utils/src/utils')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Utils import Utils
from scipy.interpolate import Rbf
from matplotlib import colors


class ViewResultsUtils:
    def __init__(self):
        # x, y, z = 1D array
        # t = 1D array. eg; 0, 0.01, 0.02, 0.03, 0.04
        # u: length of u = no_pt * t

        self.x = None
        self.y = None
        self.z = None
        self.Xi = None
        self.Yi = None
        self.no_pt = None
        self.max_x = None
        self.max_y = None

        self.V = None
        self.v = None
        self.t = None
        self.ut = Utils()

    def assign_no_pt(self, no_pt):
        self.no_pt = no_pt
        return

    def assign_x_y_z(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.max_x, self.max_y = round(x.max()), round(y.max())
        self.Xi, self.Yi = self.generate_meshgrid()
        return

    def assign_V_v_t(self, V, v, t):
        self.V = np.reshape(V, [t.shape[0], self.no_pt])
        self.v = np.reshape(v, [t.shape[0], self.no_pt])
        self.t = t
        return

    def show_pattern_V(self, start_time, end_time, no_of_plot):
        dt = self.t[2] - self.t[1]

        sqrt_no_of_plot = int(np.sqrt(no_of_plot))
        start_time_step, end_time_step = int(start_time / dt), int(end_time / dt) + 1
        step_plot = int((end_time_step - start_time_step + 1) // no_of_plot)
        V_specific_period = self.get_V_specific_period(start_time, end_time)

        fig = plt.figure()  #figsize=(10, 10)
        for i in range(no_of_plot):
            print(i)
            ax = fig.add_subplot(sqrt_no_of_plot, sqrt_no_of_plot, i + 1)
            voltage_to_plot = V_specific_period[i * step_plot]
            # cbar = self.plot_scatter(ax, voltage_to_plot)
            voltage_to_plot_grid = self.linear_interp(voltage_to_plot)
            cbar = self.plot_imshow(ax, voltage_to_plot_grid)

            ax.set_title('t={:.2f}'.format(self.t[i * step_plot] + start_time), fontsize=20)
            fig.colorbar(cbar, ax=ax)

        plt.tight_layout()

    def plot_imshow(self, ax, voltage_to_plot):
        VMIN, VMAX = 0, np.max(self.V)
        cbar = ax.imshow(voltage_to_plot, vmin=VMIN, vmax=VMAX, extent=[0, self.max_x, 0, self.max_y])
        return cbar

    def plot_scatter(self, ax, voltage_to_plot):
        VMIN, VMAX = 0, np.max(self.V)
        cbar = ax.scatter(self.x, self.y, c=voltage_to_plot, s=1, vmin=VMIN, vmax=VMAX)
        ax.axis('equal')
        return cbar

    def generate_meshgrid(self):
        x, y, z = self.x.copy(), self.y.copy(), self.z.copy()
        Xi = np.arange(round(x.min()), round(x.max() + 2), 2)
        Yi = np.arange(round(y.min()), round(y.max() + 2), 2)
        Xi, Yi = np.meshgrid(Xi, Yi)
        return Xi, Yi

    def linear_interp(self, voltage_to_plot):
        assert np.ndim(voltage_to_plot) == 1, 'voltage_to_plot must be 1D array'
        x, y, z = self.x.copy(), self.y.copy(), self.z.copy()
        Xi, Yi = self.Xi.copy(), self.Yi.copy()

        rbf = Rbf(x, y, voltage_to_plot, function='linear', smooth=3)
        ai = rbf(Xi, Yi)
        return ai

    def get_V_specific_period(self, start_time, end_time):
        dt = self.t[2] - self.t[1]
        start_time_step, end_time_step = int(start_time / dt), int(end_time / dt) + 1
        return self.V[start_time_step:end_time_step].copy()

    def get_v_specific_period(self, start_time, end_time):
        dt = self.t[2] - self.t[1]
        start_time_step, end_time_step = int(start_time / dt), int(end_time / dt) + 1
        return self.v[start_time_step:end_time_step].copy()

    def get_t_specific_period(self, start_time, end_time):
        dt = self.t[2] - self.t[1]
        start_time_step, end_time_step = int(start_time / dt), int(end_time / dt) + 1
        return self.t[start_time_step:end_time_step].copy()

    def get_index(self, region):
        index = []
        coord = np.zeros([self.no_pt, 3])
        coord[:, 0], coord[:, 1] = self.x.copy(), self.y.copy()
        for interest_region in region:
            index_tmp = self.ut.find_nearest(coord, interest_region)
            # index_tmp = np.where((interest_region == coord).all(axis=1))
            # index_tmp = np.squeeze(index_tmp[0])
            index.append(index_tmp)
        index = np.array(index)
        return index

    def get_V_v_t_at_specific_time(self, start_time, end_time):
        V_specific_period = self.get_V_specific_period(start_time, end_time)
        v_specific_period = self.get_v_specific_period(start_time, end_time)
        t = self.get_t_specific_period(start_time, end_time)
        return V_specific_period, v_specific_period, t

    def show_V_v_at_specific_points_sphere(self, interest_pt_index, legend, start_time, end_time):
        V_specific_period, v_specific_period, t = self.get_V_v_t_at_specific_time(start_time, end_time)
        numb = 0
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        for idx in interest_pt_index:
            ax1.plot(t, V_specific_period[:, idx], label='\u03F4={0:0.2f}'.format(legend[numb]))
            ax2.plot(t, v_specific_period[:, idx], label='\u03F4={0:0.2f}'.format(legend[numb]))
            numb += 1
        ax1.set_xlabel('time (unit)')
        ax1.set_title('V (fast var)')
        ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax2.set_xlabel('time (unit)')
        ax2.set_title('v (slow var)')
        ax2.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout()
        # plt.show()
        return

    def show_V_v_at_specific_points_grid(self, fig, interest_pt_index, start_time, end_time):
        V_specific_period, v_specific_period, t = self.get_V_v_t_at_specific_time(start_time, end_time)
        numb = 0
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        for idx in interest_pt_index:
            ax1.plot(t, V_specific_period[:, idx], label='{}'.format(numb))
            ax2.plot(t, v_specific_period[:, idx], label='{}'.format(numb))
            numb += 1
        ax1.set_xlabel('time (unit)')
        ax1.set_title('V (fast var)')
        ax1.legend()
        ax2.set_xlabel('time (unit)')
        ax2.set_title('v (slow var)')
        ax2.legend()
        plt.tight_layout()
        return

    def plot_phase_diagram(self, fig,  a, delta, gamma, stimulated_current):
        V = np.linspace(-0.2, 1.2, a.shape[0])
        v = np.linspace(-0.1, 0.2, a.shape[0])
        # V = np.linspace(-0.2, 1.2, 100)
        # v = np.linspace(-0.1, 0.2, 100)

        to_plot_V = ((a - V) * (V - 1) * V) + stimulated_current
        to_plot_v = (v + delta) / gamma
        ax = fig.add_subplot(111)
        ax.plot(V, to_plot_V)
        ax.plot(v, to_plot_v)
        ax.set_xlabel('fast var')
        ax.set_ylabel('slow var')
        # plt.show()
        return

    def plot_trajectory(self, fig, interest_pt_index):
        V_to_plot = self.V.copy()
        v_to_plot = self.v.copy()
        ax = fig.add_subplot(111)
        ax.plot(V_to_plot[:, interest_pt_index], v_to_plot[:, interest_pt_index])
        ax.grid()
        return

    def plot_U(self, time, U_update):
        plt.figure()
        plt.plot(time, U_update)
        plt.grid()
        plt.show()
        return

    def plot_theta_phi_V(self, start_time, end_time, no_of_plot):
        dt = self.t[2] - self.t[1]

        sqrt_no_of_plot = int(np.sqrt(no_of_plot))
        start_time_step, end_time_step = int(start_time / dt), int(end_time / dt) + 1
        step_plot = int((end_time_step - start_time_step + 1) // no_of_plot)
        V_specific_period = self.get_V_specific_period(start_time, end_time)

        coord = np.array([self.x, self.y, self.z]).transpose()
        _, phi, theta = self.ut.xyz2sph(coord)

        thetaI = np.linspace(theta.min(), theta.max())
        phiI = np.linspace(phi.min(), phi.max())
        thetaI, phiI = np.meshgrid(thetaI, phiI)

        fig = plt.figure()  #figsize=(10, 10)
        for i in range(no_of_plot):
            print(i)
            ax = fig.add_subplot(sqrt_no_of_plot, sqrt_no_of_plot, i + 1)
            ax.set_xlabel('\u03F4')  #THETA
            ax.set_ylabel('\u03C6')  #PHI

            V_tmp = V_specific_period[i * step_plot]

            # === plot raw ===
            # cbar = ax.scatter(theta, phi, c=V_tmp)
            # === plot smooth before plotting ====
            rbf = Rbf(theta, phi, V_tmp, function='linear', smooth=1)
            V_I = rbf(thetaI, phiI)
            cbar = ax.imshow(V_I, origin='lower', extent=[thetaI.min(), thetaI.max(), phiI.min(), phiI.max()])

            ax.set_title('t={:.2f}'.format(self.t[i * step_plot]), fontsize=20)
            fig.colorbar(cbar, ax=ax)

        plt.tight_layout()
        # plt.show()
        return

    def plot_temporal_V(self):
        ut = Utils()
        coord = np.array([self.x, self.y, self.z]).transpose()
        _, phi, theta = ut.xyz2sph(coord)

        thetaI = np.linspace(theta.min(), theta.max())
        phiI = np.linspace(phi.min(), phi.max())
        thetaI, phiI = np.meshgrid(thetaI, phiI)
        from scipy.interpolate import Rbf
        dt = self.t[2] - self.t[1]
        fig = plt.figure(1)
        # fig2 = plt.figure(2)
        # fig3 = plt.figure(3)
        ax = fig.add_subplot(111)
        # ax2 = fig2.add_subplot(111)
        # ax3= fig3.add_subplot(111)
        skip_dt = 100
        skip_time_step = int(skip_dt * (self.t.shape[0] - 1) / self.t[-1])

        for n in range(0, self.t.shape[0], skip_time_step):
            print(n * dt)
            V_tmp = self.V[n]
            rbf = Rbf(theta, phi, V_tmp, function='linear', smooth=1)
            V_I = rbf(thetaI, phiI)
            ax.plot(thetaI[0], V_I[0], label='time:{}'.format(n * dt))
            # ax2.imshow(V_I, origin='lower', extent=[thetaI.min(), thetaI.max(), phiI.min(), phiI.max()])
            # ax3.scatter(theta, phi, c=V_tmp)
        ax.set_xlabel('theta')
        ax.set_ylabel('V')
        ax.set_title('Fast Variable (V)')
        ax.grid()
        ax.legend(loc='center lower', bbox_to_anchor=(1.0, -0.15), ncol=4)
        fig.tight_layout()
        return

    def plot_heart_3Dview_V(self, start_time, end_time, no_of_plot):
        dt = self.t[2] - self.t[1]
        VMIN, VMAX = 0, np.max(self.V)

        sqrt_no_of_plot = int(np.sqrt(no_of_plot))
        start_time_step, end_time_step = int(start_time / dt), int(end_time / dt) + 1
        step_plot = int((end_time_step - start_time_step + 1) // no_of_plot)
        V_specific_period = self.get_V_specific_period(start_time, end_time)

        coord = np.array([self.x, self.y, self.z]).transpose()
        coord = coord - coord.mean(axis=0)
        _, phi, theta = self.ut.xyz2sph(coord)

        for i in range(no_of_plot):
            print(i)
            fig = plt.figure()
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')

            V_tmp = V_specific_period[i * step_plot]

            # === plot raw ===
            cbar1 = ax1.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=V_tmp,
                                vmin=VMIN, vmax=VMAX)
            ax1.set_title('t={:.2f}'.format(self.t[i * step_plot]), fontsize=20)
            fig.colorbar(cbar1, ax=ax1)

        fig.tight_layout()
        # plt.show()
        return

    def plot_heart_diff_sph_view_V(self, start_time, end_time, no_of_plot):
        dt = self.t[2] - self.t[1]
        VMIN, VMAX = 0, np.max(self.V)

        sqrt_no_of_plot = int(np.sqrt(no_of_plot))
        start_time_step, end_time_step = int(start_time / dt), int(end_time / dt) + 1
        step_plot = int((end_time_step - start_time_step + 1) // no_of_plot)
        V_specific_period = self.get_V_specific_period(start_time, end_time)

        coord = np.array([self.x, self.y, self.z]).transpose()
        cog = coord.mean(axis=0)
        r, phi, theta = self.ut.xyz2sph(coord - cog)

        thetaI = np.linspace(theta.min(), theta.max())
        phiI = np.linspace(phi.min(), phi.max())
        rI = np.linspace(r.min(), r.max())
        thetaI, phiI = np.meshgrid(thetaI, phiI)

        # fig1 = plt.figure(7)
        # fig2 = plt.figure(5)  # figsize=(10, 10)
        fig3 = plt.figure(8)  # figsize=(10, 10)

        for i in range(no_of_plot):
            # ax1 = fig1.add_subplot(sqrt_no_of_plot, sqrt_no_of_plot, i + 1)
            # ax1.set_xlabel('r')
            # ax1.set_ylabel('\u03C6')  #PHI
            #
            # ax2 = fig2.add_subplot(sqrt_no_of_plot, sqrt_no_of_plot, i + 1)
            # ax2.set_xlabel('r')
            # ax2.set_ylabel('\u03F4')  #THETA

            ax3 = fig3.add_subplot(sqrt_no_of_plot, sqrt_no_of_plot, i + 1)
            ax3.set_xlabel('\u03F4')  #THETA
            ax3.set_ylabel('\u03C6')  #PHI

            V_tmp = V_specific_period[i * step_plot]

            # # === plot raw ===
            # cbar1 = ax1.scatter(r, phi, c=V_tmp, s=10, vmin=VMIN, vmax=VMAX)
            # cbar2 = ax2.scatter(r, theta, c=V_tmp, s=10, vmin=VMIN, vmax=VMAX)
            # cbar3 = ax3.scatter(theta, phi, c=V_tmp, s=10, vmin=VMIN, vmax=VMAX)
            # #=== plot smooth before plotting ====
            # rbf = Rbf(r, phi, V_tmp, function='linear', smooth=1)
            # V_I = rbf(thetaI, phiI)
            # cbar1 = ax3.imshow(V_I, origin='lower', extent=[r.min(), r.max(), phiI.min(), phiI.max()])
            #
            # rbf = Rbf(r, theta, V_tmp, function='linear', smooth=1)
            # V_I = rbf(thetaI, phiI)
            # cbar2 = ax3.imshow(V_I, origin='lower', extent=[r.min(), r.max(), thetaI.min(), thetaI.max()])
            #
            rbf = Rbf(theta, phi, V_tmp, function='linear', smooth=1)
            V_I = rbf(thetaI, phiI)
            cbar3 = ax3.imshow(V_I, origin='lower', extent=[thetaI.min(), thetaI.max(), phiI.min(), phiI.max()],
                               vmin=VMIN, vmax=VMAX)
            # ================================

            # ax1.set_title('t={:.2f}'.format(self.t[i * step_plot] + start_time), fontsize=20)
            # fig1.colorbar(cbar1, ax=ax1)
            # ax2.set_title('t={:.2f}'.format(self.t[i * step_plot] + start_time), fontsize=20)
            # fig2.colorbar(cbar2, ax=ax2)
            ax3.set_title('t={:.2f}'.format(self.t[i * step_plot] + start_time), fontsize=20)
            fig3.colorbar(cbar3, ax=ax3)

        # fig1.tight_layout()
        # fig2.tight_layout()
        fig3.tight_layout()
        # plt.show()
        return

    def plot_heart_smooth_V_subplotimage(self, start_time, end_time, no_of_plot):
        dt = self.t[2] - self.t[1]
        VMIN, VMAX = 0, np.max(self.V)

        sqrt_no_of_plot = int(np.sqrt(no_of_plot))
        start_time_step, end_time_step = int(start_time / dt), int(end_time / dt) + 1
        step_plot = int((end_time_step - start_time_step + 1) // no_of_plot)
        V_specific_period = self.get_V_specific_period(start_time, end_time)

        coord = np.array([self.x, self.y, self.z]).transpose()
        coord = coord - coord.mean(axis=0)

        fig1 = plt.figure(1, figsize=(10, 8))
        # fig2 = plt.figure(2, figsize=(10, 8))
        # fig3 = plt.figure(3, figsize=(10, 8))
        # fig4 = plt.figure(4, figsize=(10, 8))
        fig1.suptitle('plot_surface', fontsize=16)
        # fig2.suptitle('raw', fontsize=16)
        # fig3.suptitle('meshgrid_point', fontsize=16)
        # fig4.suptitle('raw_with_surface', fontsize=16)
        for i in range(no_of_plot):
            V_tmp = V_specific_period[i * step_plot]

            XI, YI, ZI, colors_meshgrid = self.sampling_from_data_to_create_meshgrid(coord, V_tmp)
            rgb = self.assign_value_to_cmap(colors_meshgrid)

            ax1 = fig1.add_subplot(sqrt_no_of_plot, sqrt_no_of_plot, i + 1, projection='3d')
            # ax2 = fig2.add_subplot(sqrt_no_of_plot, sqrt_no_of_plot, i + 1, projection='3d')
            # ax3 = fig3.add_subplot(sqrt_no_of_plot, sqrt_no_of_plot, i + 1, projection='3d')
            # ax4 = fig4.add_subplot(sqrt_no_of_plot, sqrt_no_of_plot, i + 1, projection='3d')

            ax1.set_title('t={:.2f}'.format(self.t[i * step_plot] + start_time), fontsize=14)
            # ax2.set_title('t={:.2f}'.format(self.t[i * step_plot] + start_time), fontsize=14)
            # ax3.set_title('t={:.2f}'.format(self.t[i * step_plot] + start_time), fontsize=14)
            # ax4.set_title('t={:.2f}'.format(self.t[i * step_plot] + start_time), fontsize=14)

            # ax1.view_init(elev=24, azim=-23)
            # ax2.view_init(elev=17, azim=17)
            # ax3.view_init(elev=17, azim=17)
            # ax4.view_init(elev=17, azim=17)

            # if only want wireframe, change shade=False
            cbar1 = ax1.plot_surface(XI, YI, ZI, facecolors=rgb, shade=True, vmin=VMIN, vmax=VMAX)
            # surf.set_facecolor((0, 0, 0, 0))  # if only want wireframe, run tjis and change shade=False
            # cbar2 = ax2.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=V_tmp, vmin=VMIN, vmax=VMAX)
            # cbar3 = ax3.scatter(XI, YI, ZI, c=colors_meshgrid.flatten(), vmin=VMIN, vmax=VMAX)
            # ax4.plot_surface(XI, YI, ZI, color='0.5')
            # cbar4 = ax4.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=V_tmp, vmin=VMIN, vmax=VMAX)

            fig1.colorbar(cbar1, ax=ax1)
            # fig2.colorbar(cbar2, ax=ax2)
            # fig3.colorbar(cbar3, ax=ax3)
            # fig4.colorbar(cbar4, ax=ax4)
        # plt.show()
        return

    def get_frame_heart_smooth_V(self, start_time, end_time, no_of_plot):
        dt = self.t[2] - self.t[1]
        VMIN, VMAX = 0, np.max(self.V)

        start_time_step, end_time_step = int(start_time / dt), int(end_time / dt) + 1
        step_plot = int((end_time_step - start_time_step + 1) // no_of_plot)
        V_specific_period = self.get_V_specific_period(start_time, end_time)

        coord = np.array([self.x, self.y, self.z]).transpose()
        coord = coord - coord.mean(axis=0)

        frame = []
        for i in range(no_of_plot):
            print(i)
            V_tmp = V_specific_period[i * step_plot]
            XI, YI, ZI, colors_meshgrid = self.sampling_from_data_to_create_meshgrid(coord, V_tmp)
            rgb = self.assign_value_to_cmap(colors_meshgrid)

            fig1 = plt.figure()  #figsize=(10, 8)
            ax1 = fig1.add_subplot(111, projection='3d')
            ax1.set_title('t={:.2f}'.format(self.t[i * step_plot + start_time]), fontsize=14)
            # if only want wireframe, change shade=False
            cbar1 = ax1.plot_surface(XI, YI, ZI, facecolors=rgb, shade=True, vmin=VMIN, vmax=VMAX)
            fig1.colorbar(cbar1, ax=ax1)

            fig1.canvas.draw()
            image = np.frombuffer(fig1.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
            frame.append(image)
            plt.close()
        frames_list = np.array(frame)
        return frames_list

    def get_frame_heart_raw_V(self, start_time, end_time, no_of_plot):
        dt = self.t[2] - self.t[1]
        VMIN, VMAX = 0, np.max(self.V)

        start_time_step, end_time_step = int(start_time / dt), int(end_time / dt) + 1
        step_plot = int((end_time_step - start_time_step + 1) // no_of_plot)
        V_specific_period = self.get_V_specific_period(start_time, end_time)

        coord = np.array([self.x, self.y, self.z]).transpose()
        coord = coord - coord.mean(axis=0)

        frame = []
        for i in range(no_of_plot):
            print(i)
            V_tmp = V_specific_period[i * step_plot]
            XI, YI, ZI, colors_meshgrid = self.sampling_from_data_to_create_meshgrid(coord, V_tmp)

            fig1 = plt.figure()  #figsize=(10, 8)
            ax1 = fig1.add_subplot(111, projection='3d')
            ax1.set_title('t={:.2f}'.format(self.t[i * step_plot + start_time]), fontsize=14)
            ax1.plot_surface(XI, YI, ZI, color='0.5')
            cbar1 = ax1.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=V_tmp, vmin=VMIN, vmax=VMAX)
            fig1.colorbar(cbar1, ax=ax1)

            fig1.canvas.draw()
            image = np.frombuffer(fig1.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
            frame.append(image)
            plt.close()
        frames_list = np.array(frame)
        return frames_list

    def get_frame_heart_raw_V_backview(self, start_time, end_time, no_of_plot):
        dt = self.t[2] - self.t[1]
        VMIN, VMAX = 0, np.max(self.V)

        start_time_step, end_time_step = int(start_time / dt), int(end_time / dt) + 1
        step_plot = int((end_time_step - start_time_step + 1) // no_of_plot)
        V_specific_period = self.get_V_specific_period(start_time, end_time)

        coord = np.array([self.x, self.y, self.z]).transpose()
        coord = coord - coord.mean(axis=0)

        fig1 = plt.figure()  # figsize=(10, 8)
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.view_init(elev=23, azim=56)
        frame = []
        for i in range(1, no_of_plot):
            print(i)
            V_tmp = V_specific_period[i * step_plot]
            XI, YI, ZI, colors_meshgrid = self.sampling_from_data_to_create_meshgrid(coord, V_tmp)

            ax1.set_title('t={:.2f}'.format(self.t[i * step_plot + start_time]), fontsize=14)
            ax1.plot_surface(XI, YI, ZI, color='0.5')
            cbar1 = ax1.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=V_tmp, vmin=VMIN, vmax=VMAX)
            colorbar = fig1.colorbar(cbar1, ax=ax1)

            fig1.canvas.draw()
            image = np.frombuffer(fig1.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
            frame.append(image)

            ax1.clear()
            colorbar.remove()
            # plt.close()
        frames_list = np.array(frame)
        return frames_list

    def sampling_from_data_to_create_meshgrid(self, coord, color_value):
        assert np.ndim(color_value) == 1, 'color_value should be 1D array'
        assert np.ndim(coord) == 2, 'coord should be 1D array'

        gap = 5
        thetaI = np.arange(0, 360, gap) * np.pi / 180
        phiI = np.arange(0, 180, gap) * np.pi / 180
        coords_index = []
        for i in thetaI:
            for j in phiI:
                X_ = np.cos(i) * np.sin(j)
                Y_ = np.sin(i) * np.sin(j)
                Z_ = np.cos(j)
                line = np.array([X_, Y_, Z_]).reshape([1, 3])
                parall_vec_dist = coord[:, 0] * line[0, 0] + coord[:, 1] * line[0, 1] + \
                                  coord[:, 2] * line[0, 2]  # dotproduct
                n = np.argwhere(parall_vec_dist > 0).squeeze()  ## get only from those positive in direction
                parall_vec_dist = parall_vec_dist.reshape([-1, 1])
                perpd_vec = coord[n, :] - parall_vec_dist[n] * line

                perpd_dist = np.sqrt(perpd_vec[:, 0] ** 2 + perpd_vec[:, 1] ** 2 + \
                                     perpd_vec[:, 2] ** 2)

                min_rad_loc = np.argmin(perpd_dist)
                coords_index.append(n[min_rad_loc])

        coords_index = np.array(coords_index)
        coords_meshgrid = np.reshape(coord[coords_index], [phiI.shape[0], thetaI.shape[0], 3])
        colors_meshgrid = np.reshape(color_value[coords_index], [phiI.shape[0], thetaI.shape[0], 1])
        XI, YI, ZI = coords_meshgrid[:, :, 0], coords_meshgrid[:, :, 1], coords_meshgrid[:, :, 2]
        return XI, YI, ZI, colors_meshgrid

    def assign_value_to_cmap(self, color_value_meshgrid):
        dz = color_value_meshgrid.squeeze()
        offset = dz + np.abs(dz.min())
        fracs = offset.astype(float) / offset.max()
        norm = colors.Normalize(fracs.min(), fracs.max())
        rgb1 = plt.cm.viridis(norm(fracs))
        return rgb1

