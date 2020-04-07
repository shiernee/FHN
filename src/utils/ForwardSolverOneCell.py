import sys
sys.path.insert(1, '/home/sawsn/Shiernee/Utils/src/utils')

import numpy as np
import copy
from Utils import Utils
import copy


class ForwardSolverOneCell:
    def __init__(self, fhn_model):

        self.fhn_model = fhn_model
        self.ut = Utils()
        return

    def solve(self, dt, duration):

        assert self.fhn_model.V0.shape[0] == 1, 'first axis of u must be 1, indicating V0'
        assert self.fhn_model.v0.shape[0] == 1, 'first axis of u must be 1, indicating v0'

        V = np.squeeze(self.fhn_model.V0)
        v = np.squeeze(self.fhn_model.v0)

        time_pt = np.linspace(0, duration, int(duration / dt) + 1, endpoint=True, dtype='float64')

        V_update = []
        v_update = []
        V_update.append(V)
        v_update.append(v)

        for t in range(1, len(time_pt)):
            print('time: {}'.format(t * dt))
            fast_v = self.fhn_model.fast_variable(V, v)
            slow_v = self.fhn_model.slow_variable(V, v)

            applied_current = self.fhn_model.applied_current
            dVdt = fast_v + applied_current
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

            # # for debugging
            # index = np.where(([40, 40, 0] == self.point_cloud.coord).all(axis=1))
            # print('max V:{}, max_I:{}, max_dVdt:{}'.format(V[index], applied_current_at_t[index],
            #                                                dVdt[index]))
            # print('intp_V_axis1{}'.format(intp_V_axis1[index]))
            # print('intp_V_axis1_rbf{}'.format(intp_V_axis1_rbf[index]))


            V_update.append(V)
            v_update.append(v)

        V_update = np.array(V_update, dtype='float64')
        v_update = np.array(v_update, dtype='float64')

        return V_update, v_update, time_pt

