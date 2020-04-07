import sys
sys.path.insert(1, '/home/sawsn/Shiernee/FHN/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/PointCloud/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/FileIO/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/Utils/src/utils')

from FileIO import FileIO
from FHNModel import FHNModel
from ForwardSolverOneCell import ForwardSolverOneCell
from SanityCheck import SanityCheck
from Utils import Utils
from FileIO import FileIO
import numpy as np

if __name__ == '__main__':
    forward_folder = '../data/case1_one_cell/forward/'

    fileio = FileIO()
    fileio.assign_forward_folder(forward_folder)
    sc = SanityCheck()
    ut = Utils()

    # =====================================
    # D *dt /dx**2 must be < 0.5 in order for stable
    DT = 0.1
    DURATION = 700

    a = -0.1
    epsilon = 0.003
    beta = 0.5
    gamma = 1
    delta = 0.0
    APPLIED_CURRENT_VALUE = 0.0  # has to be larger than the excitability threshold, a

    # ==== initial boundary condition ===================
    V0 = np.array([[1.0]])
    v0 = np.array([[0.0]])

    # =================== FHN Model ============
    fhn_model = FHNModel(a, epsilon, beta, gamma, delta)
    fhn_model.assign_V0(V0, no_pt=1)
    fhn_model.assign_v0(v0, no_pt=1)
    fhn_model.assign_applied_current(APPLIED_CURRENT_VALUE)

    solver = ForwardSolverOneCell(fhn_model)
    V_update, v_update, time_pt = solver.solve(DT, DURATION)

    # ================================================================================== #

    fhn_model.assign_V(V_update)
    fhn_model.assign_v(v_update)
    fhn_model.assign_t(time_pt)
    fhn_model_instances = fhn_model.instance_to_dict()

    # ================ WRITE DOWN THE PARAMETER USED AND U_UPDATE INTO SAV ===================== #
    i = ut.file_number_README(forward_folder)
    fileio.write_physics_model_instance(fhn_model_instances, i, model='fhn')

    with open('{}/{}{}.txt'.format(forward_folder, 'README', i), mode='w', newline='') as csv_file:
        csv_file.write('dt={}\n'.format(DT))
        csv_file.write('simulation_duration={}\n'.format(DURATION))
        csv_file.write('a={}\n'.format(a))
        csv_file.write('epsilon={}\n'.format(epsilon))
        csv_file.write('beta={}\n'.format(beta))
        csv_file.write('gamma={}\n'.format(gamma))
        csv_file.write('delta={}\n'.format(delta))
    print('writing {}/{}{}.txt'.format(forward_folder, 'README', i))





