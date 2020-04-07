import sys
sys.path.insert(1, '/home/sawsn/Shiernee/FHN/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/FileIO/src/utils')

from ViewResultsUtils import ViewResultsUtils
from FileIO import FileIO
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss

if __name__ == '__main__':
    # case1_1D_D1_c0, case2_sphere_D1_c0, case3_2D_D1_c0,
    forward_folder = '../data/case2_2Dgrid_100/forward_6561pt/'
    START_TIME = 0
    END_TIME = 700

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
    STIMULATED_CURRENT  = np.max(applied_current)

    D = diffusion_model_instances['D']
    c = diffusion_model_instances['c']

    print('D:{}'.format(D[0]))
    print('dx:{}'.format(coord[-3:-1]))
    aa = np.array(np.where(applied_current == 0.1))
    print(coord[aa])

    x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]

    length = round(np.max(coord))
    region = [[int(length*0.5), int(length*0.5), 0],
              [int(length*0.6), int(length*0.5), 0],
              [int(length*0.7), int(length*0.5), 0],
              [int(length*0.8), int(length*0.5), 0],
              [int(length*0.9), int(length*0.5), 0]]

    # region = [[int(length * 0.5), int(length * 0.5), 0],
    #           [int(length * 0.5), int(length * 0.6), 0],
    #           [int(length * 0.5), int(length * 0.7), 0],
    #           [int(length * 0.5), int(length * 0.8), 0],
    #           [int(length * 0.5), int(length * 0.9), 0],
    #           [int(length * 0.5), int(length * 0.98), 0]]

    vr = ViewResultsUtils()
    vr.assign_x_y_z(x, y, z)
    vr.assign_no_pt(no_pt)
    vr.assign_V_v_t(V, v, t)
    no_pt = x.shape[0]

    interest_pt_index = vr.get_index(region)
    fig = plt.figure()
    vr.show_V_v_at_specific_points_grid(fig, interest_pt_index, START_TIME, END_TIME)
    fileio.save_temporalV_png_file(i)

    region = [[int(length*0.5), int(length*0.5), 0]]
    interest_pt_index = vr.get_index(region)
    fig1 = plt.figure()
    vr.plot_phase_diagram(fig1, a, delta, gamma, STIMULATED_CURRENT)
    vr.plot_trajectory(fig1, interest_pt_index)
    fileio.save_phase_diagram_png_file(i)

    # grid - 1300, scatter - forward - 2533, scatter - forward_fixIC - 2405, scatter - forward2 - 419,
    # case2_NIEDERREITER2_DATASET - 1549
    turning_pt = ss.find_peaks(V[1:, interest_pt_index[0]])[0]
    print(turning_pt)
    start, end = int(turning_pt[0]/10), int(turning_pt[-1]/10)
    vr.show_pattern_V(start_time=start, end_time=end, no_of_plot=9)
    fileio.save_spatialV_png_file(i)

    plt.show()

    # vr.show_U_at_specific_point(region=[10, 60, 0])

