import sys
sys.path.insert(1, '/home/sawsn/Shiernee/FHN/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/Diffusion/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/PointCloud/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/Utils/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/FileIO/src/utils')

import numpy as np
from FileIO import FileIO
import tensorflow as tf
from DiffusionDLModel import DiffusionDLModel
from InverseSolverFHNDiff import InverseSolverFHNDiff
from FHNDLModel import FHNDLModel


if __name__ == '__main__':

    forward_folder = '../data/case2_2Dgrid_100/forward/'
    inverse_folder = '../data/case2_2Dgrid_100/inverse/'
    i = 1  # instance_number

    BATCH_SIZE = 3000
    NUM_EPOCH = 5000
    TF_SEED = 4
    LEARNING_RATE = 0.01
    LOSS = tf.losses.MeanSquaredError()
    OPTIMIZER = tf.optimizers.Adam(lr=LEARNING_RATE)

    # ========================= get parameter from README txt file =============================== #
    fileio = FileIO()
    fileio.assign_forward_folder(forward_folder)
    fileio.assign_inverse_folder(inverse_folder)

    fhn_model_instances = fileio.read_physics_model_instance(i, 'fhn')
    t = fhn_model_instances['t']
    V = fhn_model_instances['V']
    v = fhn_model_instances['v']

    point_cloud_instances = fileio.read_point_cloud_instance(i)
    no_pt = point_cloud_instances['no_pt']
    nn_indices = point_cloud_instances['nn_indices']
    dist_intp_coord_axis1 = point_cloud_instances['dist_intp_coord_axis1']
    dist_intp_coord_axis2 = point_cloud_instances['dist_intp_coord_axis2']

    dictionary = fileio.read_forward_README_txt(i)
    DT = np.array(dictionary.get('dt'), dtype='float64')
    DURATION = np.array(dictionary.get('simulation_duration'), dtype='float64')
    INTERPOLATED_SPACING = float(dictionary.get('interpolated_spacing'))
    ORDER_ACC = int(dictionary.get('order_acc'))

    # =================== Diffusion DL Model ============
    diffusion_dl_model = DiffusionDLModel()
    diffusion_dl_model.assign_nn_indices(nn_indices)
    diffusion_dl_model.assign_interpolated_spacing(INTERPOLATED_SPACING)
    diffusion_dl_model.assign_dist_intp_coord_axis1(dist_intp_coord_axis1)
    diffusion_dl_model.assign_dist_intp_coord_axis2(dist_intp_coord_axis2)
    diffusion_dl_model.assign_t(t)
    diffusion_dl_model.assign_u(V)
    diffusion_dl_model.compute_no_pt()
    # diffusion_dl_model.compute_nn_u()

    intp_u_axis1, intp_u_axis2 = diffusion_dl_model.interpolate_u_axis1_axis2(ORDER_ACC)
    diffusion_dl_model.assign_intp_u_axis1(intp_u_axis1)
    diffusion_dl_model.assign_intp_u_axis2(intp_u_axis2)

    dudt = diffusion_dl_model.compute_dudt()
    diffusion_dl_model.assign_dudt(dudt)

    diffusion_dl_model.initialize_weight()

    # ========================= fhn model ================
    fhn_dl_model = FHNDLModel()
    fhn_dl_model.assign_t(t)
    fhn_dl_model.assign_V(V)
    fhn_dl_model.assign_v(v)
    fhn_dl_model.compute_no_pt()

    fhn_dl_model.convert_V_to_tf()
    fhn_dl_model.convert_v_to_tf()
    dVdt = fhn_dl_model.compute_dVdt()
    dvdt = fhn_dl_model.compute_dvdt()
    fhn_dl_model.assign_dVdt(dVdt)
    fhn_dl_model.assign_dvdt(dvdt)

    fhn_dl_model.initialize_weight()

    # ============== FHN Diff Model ===================
    # fhn_diff_dl_model = FHNDiffDLModel(fhn_dl_model, diffusion_dl_model)
    # fhn_diff_dl_model.assign_t(t)
    # fhn_diff_dl_model.assign_V(V)
    # fhn_diff_dl_model.assign_v(v)
    # fhn_diff_dl_model.compute_no_pt()
    #
    # fhn_diff_dl_model.assign_nn_indices(nn_indices)
    # fhn_diff_dl_model.assign_interpolated_spacing(INTERPOLATED_SPACING)
    # fhn_diff_dl_model.assign_dist_intp_coord_axis1(dist_intp_coord_axis1)
    # fhn_diff_dl_model.assign_dist_intp_coord_axis2(dist_intp_coord_axis2)
    #
    # fhn_diff_dl_model.convert_V_to_tf()
    # fhn_diff_dl_model.convert_v_to_tf()
    # dVdt = fhn_diff_dl_model.compute_dVdt()
    # dvdt = fhn_diff_dl_model.compute_dvdt()
    # fhn_diff_dl_model.assign_dVdt(dVdt)
    # fhn_diff_dl_model.assign_dvdt(dvdt)
    #
    # start_time = time.time()
    # intp_V_axis1, intp_V_axis2 = fhn_diff_dl_model.interpolate_V_axis1_axis2(ORDER_ACC)
    # fhn_diff_dl_model.assign_intp_V_axis1(intp_V_axis1)
    # fhn_diff_dl_model.assign_intp_V_axis2(intp_V_axis2)
    # print('{}'.format(time.time() - start_time))
    #
    # fhn_diff_dl_model.initialize_weight()
    # ============================================================================================ #

    inverse_solver_fhn_diffusion = InverseSolverFHNDiff(fhn_dl_model, diffusion_dl_model, ORDER_ACC)
    coeff_matrix_first_der = inverse_solver_fhn_diffusion.generate_first_der_coeff_matrix()
    coeff_matrix_second_der = inverse_solver_fhn_diffusion.generate_second_der_coeff_matrix()
    inverse_solver_fhn_diffusion.convert_coeff_matrix_to_tensor_with_correct_shape(coeff_matrix_first_der, coeff_matrix_second_der)
    n_training_epoch, training_loss = inverse_solver_fhn_diffusion.solve(NUM_EPOCH, BATCH_SIZE, LOSS, OPTIMIZER)

    # ======================================================
    fhn_dl_model.assign_training_loss(n_training_epoch, training_loss)
    fhn_dl_model_instance = fhn_dl_model.instance_to_dic()
    diffusion_dl_model_instance = diffusion_dl_model.instance_to_dic()

  # ================ WRITE DOWN THE PARAMETER USED AND WEIGHTS INTO SAV ===================== #
    i = fileio.file_number_inverse_README()
    fileio.write_inverse_physics_model_instance(fhn_dl_model_instance, i, model='fhn')
    fileio.write_inverse_physics_model_instance(diffusion_dl_model_instance, i, model='diffusion')

    with open('{}/{}{}.txt'.format(inverse_folder, 'README', i), mode='w', newline='') as csv_file:
        csv_file.write('batch_size={}\n'.format(BATCH_SIZE))
        csv_file.write('num_epoch={}\n'.format(NUM_EPOCH))
        csv_file.write('tf_seed={}\n'.format(TF_SEED))
        csv_file.write('learning_rate={}\n'.format(LEARNING_RATE))
        csv_file.write('loss_method={}\n'.format(LOSS.name))
        csv_file.write('optimizer={}'.format(OPTIMIZER.get_config()['name']))

    print('writing {}/{}{}.txt'.format(inverse_folder, 'README', i))
