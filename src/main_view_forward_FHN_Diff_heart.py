import sys
sys.path.insert(1, '/home/sawsn/Shiernee/FHN/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/FileIO/src/utils')

from ViewResultsUtils import ViewResultsUtils
from FileIO import FileIO
import matplotlib.pyplot as plt
import numpy as np
from Utils import Utils


if __name__ == '__main__':
    # case1_1D_D1_c0, case2_sphere_D1_c0, case3_2D_D1_c0,
    forward_folder = '../data/case4_LAF_21_52_36/forward/'
    START_TIME = 0
    END_TIME = 500

    fileio = FileIO()
    fileio.assign_forward_folder(forward_folder)
    i = 1
    fhn_model_instances = fileio.read_physics_model_instance(i, 'fhn')
    diffusion_model_instances = fileio.read_physics_model_instance(i, 'diffusion')
    point_cloud_instances = fileio.read_point_cloud_instance(i)

    # ========================== get variable  ================================ #
    coord = point_cloud_instances['coord']
    no_pt = point_cloud_instances['no_pt']
    t = fhn_model_instances['t']
    V = fhn_model_instances['V']
    v = fhn_model_instances['v']
    a = fhn_model_instances['a']
    delta = fhn_model_instances['delta']
    gamma = fhn_model_instances['gamma']
    applied_current = fhn_model_instances['applied_current']

    D = diffusion_model_instances['D']
    c = diffusion_model_instances['c']

    print('D:{}'.format(D[0]))
    print('dx:{}'.format(coord[-3:-1]))
    aa = np.array(np.where(applied_current == 0.1))
    print(coord[aa])

    x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]

    vr = ViewResultsUtils()
    vr.assign_x_y_z(x, y, z)
    no_pt = x.shape[0]
    vr.assign_no_pt(no_pt)
    vr.assign_V_v_t(V, v, t)

    # ================= PLOTTING =================================
    # ======= HEART RESULTS ===============
    # vr.plot_heart_smooth_V(START_TIME, end_time=END_TIME, no_of_plot=9)
    # frames_list = vr.get_frame_heart_smooth_V(START_TIME, end_time=END_TIME, no_of_plot=50)
    # fileio.write_gif_smooth_V_forward(frames_list, 'surface')
    # fileio.write_avi_smooth_V_forward(frames_list, 'surface')
    #
    # frames_list = vr.get_frame_heart_raw_V(START_TIME, end_time=END_TIME, no_of_plot=50)
    # fileio.write_gif_smooth_V_forward(frames_list, 'raw')
    # fileio.write_avi_smooth_V_forward(frames_list, 'raw')

    frames_list = vr.get_frame_heart_raw_V_backview(START_TIME, end_time=END_TIME, no_of_plot=50)
    fileio.write_gif_smooth_V_forward(frames_list, 'raw_backview')
    fileio.write_avi_smooth_V_forward(frames_list, 'raw_backview')

    fig1 = plt.figure()
    idx = 796
    vr.plot_phase_diagram(fig1, a[idx], delta[idx], gamma[idx], applied_current[idx])
    vr.plot_trajectory(fig1, idx)
    fileio.save_phase_diagram_png_file(i)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(t, V[:, idx])
    ax2.set_xlabel('time')
    ax2.set_ylabel('fast var')
    plt.grid()
    fileio.save_temporalV_png_file(i)

    plt.show()

    quit()

    # vr.plot_heart_diff_sph_view_V(START_TIME, end_time=END_TIME, no_of_plot=9)
    # fileio.save_spatialV_png_file(i)

    ut = Utils()
    coord = np.array([x, y, z]).transpose()
    r, phi, theta = ut.xyz2sph(coord)
    sph_coord = np.array([r, phi, theta]).transpose()

    theta_interest = np.linspace(0, np.pi, 5)
    sph_pt_interest = np.zeros([theta_interest.shape[0], 3])
    sph_pt_interest[:, 0] = 40
    sph_pt_interest[:, 1] = 0
    sph_pt_interest[:, 2] = theta_interest

    interest_pt_index = []
    for sph_pt_interest_tmp in sph_pt_interest:
        interest_pt_index.append(ut.find_nearest(sph_coord, sph_pt_interest_tmp))

    vr.show_V_v_at_specific_points_sphere(interest_pt_index, theta_interest, START_TIME, END_TIME)
    fileio.save_temporalV_png_file(i)


    quit()


    # ========== 2D grid results =====================
    vr.show_pattern_V(START_TIME, end_time=END_TIME, no_of_plot=9)
    fileio.save_spatialV_png_file(i)

    length = round(np.max(coord))
    region = [[int(length*0.5), int(length*0.5), 0],
              [int(length*0.6), int(length*0.5), 0],
              [int(length*0.7), int(length*0.5), 0],
              [int(length*0.8), int(length*0.5), 0],
              [int(length*0.9), int(length*0.5), 0]]

    interest_pt_index = vr.get_index(region)
    fig = plt.figure()
    vr.show_V_v_at_specific_points_grid(fig, interest_pt_index, START_TIME, END_TIME)
    fileio.save_temporalV_png_file(i)

    region = [[int(length*0.5), int(length*0.5), 0]]
    interest_pt_index = vr.get_index(region)
    fig1 = plt.figure()
    STIMULATED_CURRENT = np.max(applied_current)
    vr.plot_phase_diagram(fig1, a, delta, gamma, STIMULATED_CURRENT)
    vr.plot_trajectory(fig1, interest_pt_index)
    fileio.save_phase_diagram_png_file(i)

    plt.show()


