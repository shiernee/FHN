import sys
sys.path.insert(1, '/home/sawsn/Shiernee/FHN/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/FileIO/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/Diffusion/src/utils')

from ViewResultsUtils import ViewResultsUtils
from FileIO import FileIO
import matplotlib.pyplot as plt
import numpy as np


def print_mean_error_quantile(predicted, true_value):
    print('mean: {}, range: {} - {}'.format(predicted.mean(), predicted.min(), predicted.max()))
    print('mean \u00B1std: {} \u00B1 {}'.format(predicted.mean(), predicted.std()))
    print('error(%): {} \u00B1 {}'.format((abs(predicted - true_value) / true_value).mean() * 100,
                                          (abs(predicted - true_value) / true_value).std() * 100))
    print('quartile error(%): {} \u00B1 {}'.format(np.quantile((abs(predicted - true_value) / true_value), 0.25) * 100,
                                                   np.quantile((abs(predicted - true_value) / true_value), 0.75) * 100))


if __name__ == '__main__':
    # case1_1D_D1_c0, case2_sphere_D1_c0, case3_2D_D1_c0,
    forward_folder = '../data/case3_sphere/forward1/'
    inverse_folder = '../data/case3_sphere/inverse1/'

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
    t = fhn_model_instances['t']
    V = fhn_model_instances['V']
    v = fhn_model_instances['v']

    a = fhn_model_instances['a']
    epsilon = fhn_model_instances['epsilon']
    beta = fhn_model_instances['beta']
    gamma = fhn_model_instances['gamma']
    delta = fhn_model_instances['delta']
    applied_current = fhn_model_instances['applied_current']
    weight_a = fhn_dl_model_instances['tf_weight_a'].numpy().squeeze()
    weight_epsilon = fhn_dl_model_instances['tf_weight_epsilon'].numpy().squeeze()
    weight_beta = fhn_dl_model_instances['tf_weight_beta'].numpy().squeeze()
    weight_gamma = fhn_dl_model_instances['tf_weight_gamma'].numpy().squeeze()
    weight_delta = fhn_dl_model_instances['tf_weight_delta'].numpy().squeeze()
    weight_applied_current = fhn_dl_model_instances['tf_weight_current'].numpy().squeeze()

    D = diffusion_model_instances['D']
    c = diffusion_model_instances['c']
    weight_D = diffusion_dl_model_instances['tf_weight_D'].numpy().squeeze()
    weight_c = diffusion_dl_model_instances['tf_weight_c'].numpy().squeeze()

    training_loss = fhn_dl_model_instances['training_loss']
    n_training_epoch = fhn_dl_model_instances['n_training_epoch']

    print_mean_error_quantile(weight_a, a)
    print_mean_error_quantile(weight_epsilon, epsilon)
    print_mean_error_quantile(weight_beta, beta)
    print_mean_error_quantile(weight_gamma, gamma)
    print_mean_error_quantile(weight_delta, delta)
    print_mean_error_quantile(weight_D, D)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(n_training_epoch, training_loss)
    ax.set_xlabel('n_training_epoch')
    ax.set_ylabel('training_loss')
    ax.set_title('training_loss')

    fig1 = plt.figure(2)
    ax1 = fig1.add_subplot(111)
    ax1.plot(weight_D, '.', label='predicted D')
    ax1.plot(D, '.', label='true D')
    ax1.set_xlabel('point_id')
    ax1.set_ylabel('Value of D')
    ax1.set_title('Value of D')
    fig.tight_layout()

    fileio.save_inverse_result_file(i, model='fhn_diffusion', fig_name='loss_predicted_D')

    fig2 = plt.figure(2, figsize=[10, 10])

    ax2 = fig2.add_subplot(511)
    ax2.plot(weight_a, '.', label='predicted a')
    ax2.plot(a, '.', label='true a')
    ax2.set_xlabel('point_id')
    ax2.set_ylabel('Value of a')
    ax2.set_title('Value of a')

    ax3 = fig2.add_subplot(512)
    ax3.plot(weight_epsilon, '.', label='predicted epsilon')
    ax3.plot(epsilon, '.', label='true epsilon')
    ax3.set_xlabel('point_id')
    ax3.set_ylabel('Value of epsilon')
    ax3.set_title('Value of epsilon')

    ax4 = fig2.add_subplot(513)
    ax4.plot(weight_beta, '.', label='predicted D')
    ax4.plot(beta, '.', label='true D')
    ax4.set_xlabel('point_id')
    ax4.set_ylabel('Value of beta')
    ax4.set_title('Value of beta')

    ax5 = fig2.add_subplot(514)
    ax5.plot(weight_gamma, '.', label='predicted gamma')
    ax5.plot(gamma, '.', label='true gamma')
    ax5.set_xlabel('point_id')
    ax5.set_ylabel('Value of gamma')
    ax5.set_title('Value of gamma')

    ax6 = fig2.add_subplot(515)
    ax6.plot(weight_delta, '.', label='predicted delta')
    ax6.plot(delta, '.', label='true delta')
    ax6.set_xlabel('point_id')
    ax6.set_ylabel('Value of delta')
    ax6.set_title('Value of delta')

    fig2.tight_layout()
    # plt.show()

    fileio.save_inverse_result_file(i, model='diffusion', fig_name='fhn_weight')
