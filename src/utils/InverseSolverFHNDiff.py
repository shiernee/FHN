import sys
sys.path.insert(1, '/home/sawsn/Shiernee/Utils/src/utils')


import numpy as np
import tensorflow as tf
from Utils import Utils


class InverseSolverFHNDiff:
    def __init__(self, fhn_dl_model, diff_dl_model, order_acc):
        self.fhn_dl_model = fhn_dl_model
        self.diff_dl_model = diff_dl_model

        self.order_acc = order_acc
        self.ut = Utils()

        self.tf_coeff_matrix_first_der = None
        self.tf_coeff_matrix_second_der = None

        return

    def solve(self, num_epochs, batch_size, tf_loss, tf_optimizer):
        duration = self.fhn_dl_model.t[-1]
        dt = self.fhn_dl_model.t[1] - self.fhn_dl_model.t[0]

        tf_V = self.fhn_dl_model.tf_V.__copy__()
        tf_v = self.fhn_dl_model.tf_v.__copy__()
        tf_dVdt = self.fhn_dl_model.tf_dVdt.__copy__()
        tf_dvdt = self.fhn_dl_model.tf_dvdt.__copy__()

        tf_intp_V_axis1 = self.diff_dl_model.tf_intp_u_axis1.__copy__()
        tf_intp_V_axis2 = self.diff_dl_model.tf_intp_u_axis2.__copy__()
        tf_coeff_matrix_first_der = self.tf_coeff_matrix_first_der.__copy__()
        tf_coeff_matrix_second_der = self.tf_coeff_matrix_second_der.__copy__()

        n_training_epoch = []
        training_loss = []
        batch_iteration = int(int(duration / dt) / batch_size)

        lowest_loss_value = 1e9
        for epoch in range(num_epochs):
            loss_value = 0
            # loss_value_dVdt = 0
            # loss_value_dvdt = 0

            for iteration in range(batch_iteration):
                begin = [iteration * batch_size, 0, 0]
                size = [(iteration + 1) * batch_size, tf_intp_V_axis1.shape[1], tf_intp_V_axis1.shape[2]]
                tf_intp_V_axis1_tmp = tf.slice(tf_intp_V_axis1, begin, size)
                tf_intp_V_axis2_tmp = tf.slice(tf_intp_V_axis2, begin, size)

                begin = [iteration * batch_size, 0, 0]
                size = [(iteration + 1) * batch_size, tf_dVdt.shape[1], tf_dVdt.shape[2]]
                tf_V_tmp = tf.slice(tf_V, begin, size)
                tf_v_tmp = tf.slice(tf_v, begin, size)
                tf_dVdt_tmp = tf.slice(tf_dVdt, begin, size)
                tf_dvdt_tmp = tf.slice(tf_dvdt, begin, size)

                with tf.GradientTape(persistent=True) as tape:
                    tf_diffusion_term = self.diff_dl_model.call_dl_model(tf_intp_V_axis1_tmp, tf_intp_V_axis2_tmp,
                                                                         tf_coeff_matrix_first_der,
                                                                         tf_coeff_matrix_second_der, self.order_acc)
                    tf_fast_variable = self.fhn_dl_model.fast_variable(tf_V_tmp, tf_v_tmp)
                    tf_slow_variable = self.fhn_dl_model.slow_variable(tf_V_tmp, tf_v_tmp)

                    tf_dVdt_pred = tf_diffusion_term + tf_fast_variable + self.fhn_dl_model.tf_weight_current
                    tf_dvdt_pred = tf_slow_variable

                    pred = tf_dVdt_pred + tf_dvdt_pred
                    true = tf_dVdt_tmp + tf_dvdt_tmp

                    loss_value += tf_loss(true, pred)

                # == calculate gradient and update variables =====
                grads_diff = tape.gradient(loss_value, self.diff_dl_model.trainable_weights)
                tf_optimizer.apply_gradients(zip(grads_diff, self.diff_dl_model.trainable_weights))
                grads_fhn = tape.gradient(loss_value, self.fhn_dl_model.trainable_weights)
                tf_optimizer.apply_gradients(zip(grads_fhn, self.fhn_dl_model.trainable_weights))
                del tape

            average_loss = loss_value.numpy() / batch_iteration
            lowest_loss_value = self.check_if_avg_loss_lower_than_lowest_loss_value(lowest_loss_value,
                                                                                    average_loss)

            n_training_epoch.append(epoch)
            training_loss.append(average_loss)

            if epoch % 1 == 0:
                print('epoch {}/{}: loss{}'.format(epoch, num_epochs, average_loss))
                print('avg_weight_D: {0:0.4f}'.format(tf.reduce_mean(self.diff_dl_model.tf_weight_D).numpy()))
                print('avg_weight_a: {0:0.4f}'.format(tf.reduce_mean(self.fhn_dl_model.tf_weight_a).numpy()))
                print('avg_weight_epsilon: {0:0.4f}'.format(tf.reduce_mean(self.fhn_dl_model.tf_weight_epsilon).numpy()))
                print('avg_weight_beta: {0:0.4f}'.format(tf.reduce_mean(self.fhn_dl_model.tf_weight_beta).numpy()))
                print('avg_weight_gamma: {0:0.4f}'.format(tf.reduce_mean(self.fhn_dl_model.tf_weight_gamma).numpy()))
                print('avg_weight_delta: {0:0.4f}'.format(tf.reduce_mean(self.fhn_dl_model.tf_weight_delta).numpy()))
                print('\n')
        n_training_epoch = np.array(n_training_epoch)
        training_loss = np.array(training_loss)

        return n_training_epoch, training_loss

    def check_if_avg_loss_lower_than_lowest_loss_value(self, lowest_loss_value, current_epoch_loss_value):
        print('current_loss  = {} lowest_loss = {}'.format(current_epoch_loss_value,
                                                           lowest_loss_value))
        if current_epoch_loss_value < lowest_loss_value:
            print('assign_lowest_weight_D')
            self.diff_dl_model.assign_tf_lowest_weight_D(self.diff_dl_model.tf_weight_D)
            self.fhn_dl_model.assign_tf_lowest_weight_a(self.fhn_dl_model.tf_weight_a)
            self.fhn_dl_model.assign_tf_lowest_weight_epsilon(self.fhn_dl_model.tf_weight_epsilon)
            self.fhn_dl_model.assign_tf_lowest_weight_beta(self.fhn_dl_model.tf_weight_beta)
            self.fhn_dl_model.assign_tf_lowest_weight_gamma(self.fhn_dl_model.tf_weight_gamma)
            self.fhn_dl_model.assign_tf_lowest_weight_delta(self.fhn_dl_model.tf_weight_delta)
            lowest_loss_value = current_epoch_loss_value.copy()
        return lowest_loss_value

    def generate_first_der_coeff_matrix(self):
        coeff = self.ut.OA_coeff(self.order_acc)
        input_length = np.shape(self.diff_dl_model.tf_intp_u_axis1.numpy())[-1]
        coeff_matrix_first_der = self.ut.coeff_matrix_first_order(input_length, coeff)
        coeff_matrix_first_der = coeff_matrix_first_der.copy()
        return coeff_matrix_first_der

    def generate_second_der_coeff_matrix(self):
        coeff = self.ut.OA_coeff(self.order_acc)
        input_length2 = np.shape(self.diff_dl_model.tf_intp_u_axis2.numpy())[-1] - len(coeff) + 1
        coeff_matrix_second_der = self.ut.coeff_matrix_first_order(input_length2, coeff)
        coeff_matrix_second_der = coeff_matrix_second_der.copy()
        return coeff_matrix_second_der

    def convert_coeff_matrix_to_tensor_with_correct_shape(self, coeff_matrix_first_der, coeff_matrix_second_der):
        coeff_matrix_first_der = np.expand_dims(coeff_matrix_first_der, axis=0)  # shape(1, 5, 3)
        coeff_matrix_second_der = np.expand_dims(coeff_matrix_second_der, axis=0)  # shape(1, 3, 1)

        # coeff_matrix_first_der = np.expand_dims(coeff_matrix_first_der, axis=0)  # shape(1, 1, 5, 3)
        # coeff_matrix_second_der = np.expand_dims(coeff_matrix_second_der, axis=0)  # shape(1, 1, 3, 1)

        assert np.ndim(coeff_matrix_first_der) == 3, 'coeff_matrix_first_der must be 3D array (1, 5nn, 3)'
        assert np.ndim(coeff_matrix_first_der) == 3, 'coeff_matrix_first_der must be 3D array (1, 3nn, 1)'

        self.tf_coeff_matrix_first_der = tf.convert_to_tensor(coeff_matrix_first_der, dtype=tf.float64)
        self.tf_coeff_matrix_second_der = tf.convert_to_tensor(coeff_matrix_second_der, dtype=tf.float64)
        return

