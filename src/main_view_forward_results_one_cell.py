import sys
sys.path.insert(1, '/home/sawsn/Shiernee/FHN/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/FileIO/src/utils')

from ViewResultsUtils import ViewResultsUtils
from FileIO import FileIO
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # case1_1D_D1_c0, case2_sphere_D1_c0, case3_2D_D1_c0,
    forward_folder = '../data/case1_one_cell/forward/'
    START_TIME = 0
    END_TIME = 700

    fileio = FileIO()
    fileio.assign_forward_folder(forward_folder)
    i = 6
    fhn_model_instances = fileio.read_physics_model_instance(i, 'fhn')

    # ========================== get variable  ================================ #
    t = fhn_model_instances['t']
    V = fhn_model_instances['V']
    v = fhn_model_instances['v']
    a = fhn_model_instances['a']
    delta = fhn_model_instances['delta']
    gamma = fhn_model_instances['gamma']
    applied_current = fhn_model_instances['applied_current']
    STIMULATED_CURRENT  = np.max(applied_current)

    vr = ViewResultsUtils()
    no_pt = 1
    vr.assign_no_pt(no_pt)
    vr.assign_V_v_t(V, v, t)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.scatter(t, V)
    ax.set_xlabel('time(s)')
    ax.set_ylabel('fast variable (V)')
    ax.set_title('fast variable (V)')

    ax = fig.add_subplot(212)
    ax.scatter(t, v)
    ax.set_xlabel('time(s)')
    ax.set_ylabel('slow variable (v)')
    ax.set_title('slow variable (v)')
    fileio.save_temporalV_png_file(i)

    # fig.tight_layout()

    fig1 = plt.figure()
    V_ = np.linspace(-0.8, 1.2, t.shape[0])
    v_ = np.linspace(-0.1, 0.2, t.shape[0])
    to_plot_V = ((a - V_) * (V_ - 1) * V_) + STIMULATED_CURRENT
    to_plot_v = (v_ + delta) / gamma
    ax = fig1.add_subplot(111)
    ax.plot(V_, to_plot_V)
    ax.plot(v_, to_plot_v)
    ax.set_xlabel('fast var')
    ax.set_ylabel('slow var')
    ax.plot(V, v)
    plt.grid('on')

    fileio.save_phase_diagram_png_file(i)

    # plt.show()




