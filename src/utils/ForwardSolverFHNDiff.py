import sys
sys.path.insert(1, '/home/sawsn/Shiernee/Diffusion/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/Utils/src/utils')

import numpy as np
import copy
from Utils import Utils
import copy


class ForwardSolverFHNDiff:
    def __init__(self, point_cloud, diffusion_model, interpolated_spacing, order_acc,
                 fhn_model):

        self.fhn_model = fhn_model

        self.point_cloud = point_cloud
        self.diffusion_model = diffusion_model

        self.interpolated_spacing = interpolated_spacing
        self.order_acc = order_acc
        self.ut = Utils()

        self.coeff_matrix_first_der = None
        self.coeff_matrix_second_der = None

        self.ut = Utils()

    def solve(self, dt, duration):

        assert self.diffusion_model.nn_u0.shape[0] == 1, 'first axis of nn_u must be 1, indicating nn_u0'
        assert self.diffusion_model.u0.shape[0] == 1, 'first axis of u must be 1, indicating u0'
        assert self.fhn_model.V0.shape[0] == 1, 'first axis of u must be 1, indicating V0'
        assert self.fhn_model.v0.shape[0] == 1, 'first axis of u must be 1, indicating v0'

        coeff_matrix_first_der = self.coeff_matrix_first_der.copy()
        coeff_matrix_second_der = self.coeff_matrix_second_der.copy()
        V = np.squeeze(self.fhn_model.V0).copy()
        v = np.squeeze(self.fhn_model.v0).copy()

        time_pt = np.linspace(0, duration, int(duration / dt) + 1, endpoint=True, dtype='float64')

        V_update = []
        v_update = []
        V_update.append(V)
        v_update.append(v)

        for t in range(1, len(time_pt)):
            print('time: {}'.format(t * dt))
            # ==== FHN MODEL ====
            fast_v = self.fhn_model.fast_variable(V, v)
            slow_v = self.fhn_model.slow_variable(V, v)
            applied_current = self.fhn_model.applied_current

            # === DIFFUSION MODEL ===
            deltaD_deltaV = self.diffusion_model.del_D_delV(V, self.interpolated_spacing, self.order_acc, \
                                                            coeff_matrix_first_der, coeff_matrix_second_der)

            dVdt = deltaD_deltaV + fast_v + applied_current
            dvdt = slow_v

            next_time_pt_V = V + dt * dVdt
            next_time_pt_v = v + dt * dvdt

            if next_time_pt_V.size > 1:
                assert sum(np.isnan(next_time_pt_V)) == 0, 'at time {}, next_time_pt_V has nan'.format(t * dt)
                assert sum(np.isnan(next_time_pt_v)) == 0, 'at time {}, next_time_pt_v has nan'.format(t * dt)
                assert next_time_pt_V.shape[0] == next_time_pt_v.shape[0], 'V and v has different length'
            elif next_time_pt_V.size == 1:
                assert np.isnan(next_time_pt_V)==False, 'at time {}, next_time_pt_V has nan'.format(t * dt)
                assert np.isnan(next_time_pt_v)==False, 'at time {}, next_time_pt_v has nan'.format(t * dt)

            V = next_time_pt_V.copy()
            v = next_time_pt_v.copy()
            V_update.append(V)
            v_update.append(v)

        V_update = np.array(V_update, dtype='float64')
        v_update = np.array(v_update, dtype='float64')

        return V_update, v_update, time_pt

    def generate_first_der_coeff_matrix(self):
        coeff = self.ut.OA_coeff(self.order_acc)
        input_length = np.shape(self.point_cloud.intp_coord_axis1)[1]
        coeff_matrix_first_der = self.ut.coeff_matrix_first_order(input_length, coeff)
        self.coeff_matrix_first_der = coeff_matrix_first_der.copy()
        return

    def generate_second_der_coeff_matrix(self):
        coeff = self.ut.OA_coeff(self.order_acc)
        input_length2 = np.shape(self.point_cloud.intp_coord_axis1)[1] - len(coeff) + 1
        coeff_matrix_second_der = self.ut.coeff_matrix_first_order(input_length2, coeff)
        self.coeff_matrix_second_der = coeff_matrix_second_der.copy()
        return

