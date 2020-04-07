import sys
sys.path.insert(1, '/home/sawsn/Shiernee/Utils/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/PointCloud/src/utils')

import numpy as np
from Utils import Utils
import copy


class FHNModel():
    def __init__(self, a, epsilon, beta, gamma, delta):

        # parameter
        self.a = a  #
        self.epsilon = epsilon
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.t = None
        self.V = None
        self.v = None
        self.V0 = None
        self.v0 = None
        self.applied_current = None

        self.ut = Utils()

    def assign_a(self, a):
        self.a = a.copy()
        return

    def assign_epsilon(self, epsilon):
        self.epsilon = epsilon.copy()
        return

    def assign_beta(self, beta):
        self.beta = beta.copy()
        return

    def assign_gamma(self, gamma):
        self.gamma = gamma.copy()
        return

    def assign_delta(self, delta):
        self.delta = delta.copy()
        return

    def assign_V0(self, V0, no_pt):
        if len(V0) > 1:
            V0 = np.reshape(V0, [1, no_pt])
            assert np.ndim(V0) == 2, 'V (fast var) should be 2D array'
            self.V0 = V0.copy()
        else:
            self.V0 = np.array(V0)
        return

    def assign_v0(self, v0, no_pt):
        if len(v0) > 1:
            v0 = np.reshape(v0, [1, no_pt])
            assert np.ndim(v0) == 2, 'V (fast var) should be 2D array'
            self.v0 = v0.copy()
        else:
            self.v0 = np.array(v0)
        return

    def assign_V(self, V):
        self.V = []
        self.V = V
        return

    def assign_v(self, v):
        self.v = []
        self.v = v
        return

    def assign_t(self, time_pt):
        self.t = []
        self.t = time_pt
        return

    def assign_applied_current(self, applied_current):
        self.applied_current = applied_current
        return

    def fast_variable(self, V, v):
        a = copy.deepcopy(self.a)
        return (a - V) * (V - 1) * V - v

    def slow_variable(self, V, v):
        epsilon = copy.deepcopy(self.epsilon)
        beta = copy.deepcopy(self.beta)
        gamma = copy.deepcopy(self.gamma)
        delta = copy.deepcopy(self.delta)

        return epsilon * (beta * V - gamma * v - delta)

    def instance_to_dict(self):
        physics_model_instance = \
            {'a': self.a,
             'epsilon': self.epsilon,
             'beta': self.beta,
             'gamma': self.gamma,
             'delta': self.delta,
             't': self.t,
             'V': self.V,
             'v': self.v,
             'V0': self.V0,
             'v0': self.v0,
             'applied_current': self.applied_current,
             }

        return physics_model_instance

    def assign_read_physics_model_instances(self, physics_model_instances):
        self.a = physics_model_instances['a']
        self.epsilon = physics_model_instances['epsilon']
        self.beta = physics_model_instances['beta']
        self.gamma = physics_model_instances['gamma']
        self.delta = physics_model_instances['delta']
        self.t = physics_model_instances['t']
        self.V = physics_model_instances['V']
        self.v = physics_model_instances['v']
        self.V0 = physics_model_instances['V0']
        self.v0 = physics_model_instances['v0']
        self.applied_current = physics_model_instances['applied_current']
        print('Finish assigning read instances to Physics_Model instances')
        return






