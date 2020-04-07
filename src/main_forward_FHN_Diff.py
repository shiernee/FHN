import sys
sys.path.insert(1, '/home/sawsn/Shiernee/FHN/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/PointCloud/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/FileIO/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/Utils/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/Diffusion/src/utils')

from FHNModel import FHNModel
from ForwardSolverFHNDiff import ForwardSolverFHNDiff
from SanityCheck import SanityCheck
from Utils import Utils
from PointCloud import PointCloud
from FileIO import FileIO
import numpy as np
from DiffusionModel import DiffusionModel, BoundaryCondition


if __name__ == '__main__':
    forward_folder = '../data/case2_2Dgrid_55/forward_10201pts/'

    fileio = FileIO()
    fileio.assign_forward_folder(forward_folder)
    sc = SanityCheck()
    ut = Utils()

    # ============ SIMULATION TIME===============
    DT = 0.1
    DURATION = 1000

    # ======== set DIFFUSION MODEL parameter================
    D = 1.0
    c = 0.0
    INTERPOLATED_SPACING = 1
    ORDER_ACC = 4  # central difference by taking two elements right and left each to compute gradient
    ORDER_DERIVATIVE = 2  # [FIXED] diffusion term is second derivative
    print('\nfolder:{} \n dt: {}\n interpolated_spacing:{}\n'.format(forward_folder, DT, INTERPOLATED_SPACING))
    # ============ set FHN MODEL parameter ==========
    a = 0.1
    epsilon = 0.01
    beta = 0.5
    gamma = 1
    delta = 0.0
    APPLIED_CURRENT_VALUE = 0.0  # has to be larger than the excitability threshold, a

    # ================= read instances ==============
    instances = fileio.read_generated_instance()
    coord = instances['coord']
    region_to_apply_current = np.squeeze(instances['region_to_apply_current'])
    t = np.squeeze(instances['t'])
    V0 = np.squeeze(instances['V'])
    v0 = np.squeeze(instances['v'])
    D = np.squeeze(instances['D']) * D
    c = np.squeeze(instances['c']) * c
    a = np.squeeze(instances['a']) * a
    epsilon = np.squeeze(instances['epsilon']) * epsilon
    beta = np.squeeze(instances['beta']) * beta
    gamma = np.squeeze(instances['gamma']) * gamma
    delta = np.squeeze(instances['delta']) * delta
    applied_current = np.squeeze(instances['applied_current']) * APPLIED_CURRENT_VALUE
    local_axis1 = np.squeeze(instances['local_axis1'])
    local_axis2 = np.squeeze(instances['local_axis2'])

    # ========================= Point Cloud ==================
    point_cloud = PointCloud()
    point_cloud.assign_coord(coord, t)
    point_cloud.compute_no_pt()
    point_cloud.compute_nn_indices_neighbor(n_neighbors=8, algorithm='kd_tree')
    point_cloud.compute_nn_coord()
    point_cloud.assign_local_axes(local_axis1, local_axis2)
    # point_cloud.compute_local_axis()
    point_cloud.interpolate_coord_local_axis1_axis2(interpolated_spacing=INTERPOLATED_SPACING, order_acc=ORDER_ACC, \
                                                    order_derivative=ORDER_DERIVATIVE)
    _, dist_intp_coord_axis1, nn_indices_intp_coord_axis1 = \
        point_cloud.compute_nn_indices_neighbor_intp_coord_axis1(n_neighbors=24, algorithm='kd_tree')
    _, dist_intp_coord_axis2, nn_indices_intp_coord_axis2 = \
        point_cloud.compute_nn_indices_neighbor_intp_coord_axis2(n_neighbors=24, algorithm='kd_tree')
    dist_intp_coord_axis1 \
        = point_cloud.discard_nn_coord_out_of_radius(dist_intp_coord_axis1, radius=3)
    dist_intp_coord_axis2 \
        = point_cloud.discard_nn_coord_out_of_radius(dist_intp_coord_axis2, radius=3)
    point_cloud.assign_dist_intp_coord_axis12(dist_intp_coord_axis1, dist_intp_coord_axis2)
    point_cloud.assign_nn_indices_intp_coord_axis12(nn_indices_intp_coord_axis1, nn_indices_intp_coord_axis2)
    # point_cloud.compute_distance_intp_coord_axis12_to_ori_coord()
    # coord_DC, D = fileio.read_D_file_csv(D_file_csv)
    # sc.check_if_equal_coordA_coordB(coord, coord_DC)

    # =================== FHN Model ============
    fhn_model = FHNModel(a, epsilon, beta, gamma, delta)
    fhn_model.assign_V0(V0, point_cloud.no_pt)
    fhn_model.assign_v0(v0, point_cloud.no_pt)
    fhn_model.assign_applied_current(applied_current)

    # == Boundary Condition only apply for 2D cases ==
    # get the border of 2D grid]
    bc_region_2D = [0.01, 0.99*point_cloud.coord[:, 0].max(), 0.01, 0.99*point_cloud.coord[:, 1].max()]
    bc = BoundaryCondition(point_cloud, bc_region_2D)
    bc.set_bc_type('neumann')


    # =========== DIFFUSION Model ===============
    physics_model = DiffusionModel()
    physics_model.assign_point_cloud_object(point_cloud)
    physics_model.assign_u0(V0)
    physics_model.assign_D_c(D, c)
    physics_model.compute_nn_u0()
    physics_model.compute_nn_D()

    intp_D_axis1 = physics_model.interpolate_D(point_cloud.dist_intp_coord_axis1,
                                               point_cloud.nn_indices_intp_coord_axis1, ORDER_ACC)
    intp_D_axis2 = physics_model.interpolate_D(point_cloud.dist_intp_coord_axis2,
                                               point_cloud.nn_indices_intp_coord_axis2, ORDER_ACC)
    physics_model.assign_intp_D_axis1(intp_D_axis1)
    physics_model.assign_intp_D_axis2(intp_D_axis2)
    physics_model.assign_boundary_condition(bc)

    # ======================================================================= #

    solver = ForwardSolverFHNDiff(point_cloud, physics_model, INTERPOLATED_SPACING, ORDER_ACC,fhn_model)
    solver.generate_first_der_coeff_matrix()
    solver.generate_second_der_coeff_matrix()
    V_update, v_update, time_pt = solver.solve(DT, DURATION)

    # ================================================================================== #

    fhn_model.assign_V(V_update)
    fhn_model.assign_v(v_update)
    fhn_model.assign_t(time_pt)
    fhn_model_instances = fhn_model.instance_to_dict()

    physics_model.assign_u_update(V_update)
    physics_model.assign_t(time_pt)
    physics_model.compute_nn_u()
    physics_model_instances = physics_model.instance_to_dict()

    point_cloud_instances = point_cloud.instance_to_dict()
    # print('D:{}'.format(D))

    # ================ WRITE DOWN THE PARAMETER USED AND U_UPDATE INTO SAV ===================== #
    i = ut.file_number_README(forward_folder)
    fileio.write_physics_model_instance(fhn_model_instances, i, 'fhn')
    fileio.write_physics_model_instance(physics_model_instances, i, 'diffusion')
    fileio.write_point_cloud_instance(point_cloud_instances, i)

    with open('{}/{}{}.txt'.format(forward_folder, 'README', i), mode='w', newline='') as csv_file:
        csv_file.write('dt={}\n'.format(DT))
        csv_file.write('simulation_duration={}\n'.format(DURATION))
        csv_file.write('interpolated_spacing={}\n'.format(INTERPOLATED_SPACING))
        csv_file.write('order_acc={}\n'.format(ORDER_ACC))
        csv_file.write('a={}\n'.format(a))
        csv_file.write('epsilon={}\n'.format(epsilon))
        csv_file.write('beta={}\n'.format(beta))
        csv_file.write('gamma={}\n'.format(gamma))
        csv_file.write('delta={}\n'.format(delta))
        csv_file.write('D={}\n'.format(D))
        csv_file.write('c={}\n'.format(c))
    print('writing {}/{}{}.txt'.format(forward_folder, 'README', i))





