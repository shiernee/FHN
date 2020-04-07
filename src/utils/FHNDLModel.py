import tensorflow as tf
import numpy as np


class FHNDLModel(tf.keras.Model):
    def __init__(self):
        super(FHNDLModel, self).__init__()

        self.t = None
        self.V = None
        self.v = None
        self.no_pt = None

        self.tf_weight_a = None
        self.tf_weight_epsilon = None
        self.tf_weight_beta = None
        self.tf_weight_gamma = None
        self.tf_weight_delta = None
        self.tf_weight_current = None

        self.tf_lowest_weight_a = None
        self.tf_lowest_weight_epsilon = None
        self.tf_lowest_weight_beta = None
        self.tf_lowest_weight_gamma = None
        self.tf_lowest_weight_delta = None
        self.tf_lowest_weight_current = None

        self.tf_V = None
        self.tf_v = None
        self.tf_dVdt = None
        self.tf_dvdt = None

        self.training_loss = None
        self.n_training_epoch = None

    def assign_t(self, t):
        self.t = t
        return

    def assign_V(self, V):
        self.V = V.copy()
        return

    def assign_v(self, v):
        self.v = v.copy()
        return

    def convert_V_to_tf(self):
        assert np.ndim(self.V) == 2, 'u must be 2D array (time_pt, no_pt)'
        V = self.V[:-1].copy()  # take till second last only, bcz last point does not have dVdt
        V = np.expand_dims(V, axis=-1)  # shape(time_pt, no_pt, 1)
        self.tf_V = tf.convert_to_tensor(V, dtype=tf.float64)
        return

    def convert_v_to_tf(self):
        assert np.ndim(self.v) == 2, 'u must be 2D array (time_pt, no_pt)'
        v = self.v[:-1].copy() #take till second last only, bcz last point does not have dvdt
        v = np.expand_dims(v, axis=-1)  # shape(time_pt, no_pt, 1)
        self.tf_v = tf.convert_to_tensor(v, dtype=tf.float64)
        return

    def compute_no_pt(self):
        self.no_pt = self.V.shape[-1]
        return

    def assign_tf_lowest_weight_a(self, tf_lowest_weight_a):
        self.tf_lowest_weight_a = tf_lowest_weight_a.__copy__()
        return

    def assign_tf_lowest_weight_epsilon(self, tf_lowest_weight_epsilon):
        self.tf_lowest_weight_epsilon = tf_lowest_weight_epsilon.__copy__()
        return

    def assign_tf_lowest_weight_beta(self, tf_lowest_weight_beta):
        self.tf_lowest_weight_beta = tf_lowest_weight_beta.__copy__()
        return

    def assign_tf_lowest_weight_gamma(self, tf_lowest_weight_gamma):
        self.tf_lowest_weight_gamma = tf_lowest_weight_gamma.__copy__()
        return

    def assign_tf_lowest_weight_delta(self, tf_lowest_weight_delta):
        self.tf_lowest_weight_delta = tf_lowest_weight_delta.__copy__()
        return

    def initialize_weight(self):
        self.tf_weight_a = tf.Variable(tf.random.normal([1, self.no_pt, 1], mean=-0.1, stddev=0.1,
                                                        dtype=tf.float64), trainable=True, name='a')
        # self.tf_weight_a.assign(np.ones(self.tf_weight_a.shape) * -0.1)
        # =============================
        self.tf_weight_epsilon = tf.Variable(tf.random.normal([1, self.no_pt, 1], mean=0.01, stddev=0.001,
                                                              dtype=tf.float64), trainable=True, name='epsilon')
        # self.tf_weight_epsilon.assign(np.ones(self.tf_weight_epsilon.shape) * 0.01)
        # =============================
        self.tf_weight_beta = tf.Variable(tf.random.normal([1, self.no_pt, 1], mean=0.5, stddev=0.1,
                                                           dtype=tf.float64), trainable=True, name='beta')
        # self.tf_weight_beta.assign(np.ones(self.tf_weight_beta.shape) * 0.5)
        # =============================
        self.tf_weight_gamma = tf.Variable(tf.random.normal([1, self.no_pt, 1], mean=1.0, stddev=0.1,
                                                            dtype=tf.float64), trainable=True, name='gamma')
        # self.tf_weight_gamma.assign(np.ones(self.tf_weight_gamma.shape) * 1.0)
        # =============================
        self.tf_weight_delta = tf.Variable(tf.random.normal([1, self.no_pt, 1], mean=0.0, stddev=0.001,
                                                            dtype=tf.float64), trainable=True, name='delta')
        # self.tf_weight_delta.assign(np.ones(self.tf_weight_delta.shape) * 0.0)
        # =============================
        self.tf_weight_current = tf.Variable(tf.random.normal([1, self.no_pt, 1], mean=0.0, stddev=0.001,
                                                              dtype=tf.float64), trainable=False, name='current')
        self.tf_weight_current.assign(np.zeros(self.tf_weight_current.shape))

        return

    def compute_dVdt(self):
        assert np.ndim(self.V) == 2, 'u must be 2D array (time_pt, no_pt)'

        dVdt = []
        V = self.V.copy()
        dt = self.t[1] - self.t[0]
        for n in range(len(V) - 1):
            tmp = np.array(V[n + 1], dtype='float64') - np.array(V[n], dtype='float64')
            dVdt.append(tmp / dt)

        dVdt = np.array(dVdt, dtype='float64')
        dVdt = np.expand_dims(dVdt, -1)  # shape (time_pt, no_pt, 1)
        # dVdt = np.expand_dims(dVdt, -1)  #shape (time_pt, no_pt, 1, 1)
        return dVdt

    def compute_dvdt(self):
        assert np.ndim(self.v) == 2, 'u must be 2D array (time_pt, no_pt)'

        dvdt = []
        v = self.v.copy()
        dt = self.t[1] - self.t[0]
        for n in range(len(v) - 1):
            tmp = np.array(v[n + 1], dtype='float64') - np.array(v[n], dtype='float64')
            dvdt.append(tmp / dt)

        dvdt = np.array(dvdt, dtype='float64')
        dvdt = np.expand_dims(dvdt, -1)  # shape (time_pt, no_pt, 1)
        # dudt = np.expand_dims(dvdt, -1)  #shape (time_pt, no_pt, 1, 1)
        return dvdt

    def assign_dVdt(self, dVdt_true):
        self.tf_dVdt = tf.convert_to_tensor(dVdt_true, dtype=tf.float64)
        return

    def assign_dvdt(self, dvdt_true):
        self.tf_dvdt = tf.convert_to_tensor(dvdt_true, dtype=tf.float64)
        return

    # def call_dl_model(self, tf_V, tf_v):
    #     tf_fast_variable = self.fast_variable(tf_V, tf_v)
    #     tf_slow_variable = self.slow_variable(tf_V, tf_v)
    #     return tf_fast_variable, tf_slow_variable

    def fast_variable(self, tf_V, tf_v):
        return (self.tf_weight_a - tf_V) * (tf_V - 1) * tf_V - tf_v

    def slow_variable(self, tf_V, tf_v):
        return self.tf_weight_epsilon * \
               (self.tf_weight_beta * tf_V - self.tf_weight_gamma * tf_v - self.tf_weight_delta)

    def assign_training_loss(self, n_training_epoch, training_loss):
        self.n_training_epoch = n_training_epoch
        self.training_loss = training_loss
        return

    def instance_to_dic(self):
        fhn_dl_model_instance = \
            {'t': self.t,
             'V': self.V,
             'v': self.v,
             'no_pt': self.no_pt,

             'tf_weight_a': self.tf_weight_a,
             'tf_weight_epsilon': self.tf_weight_epsilon,
             'tf_weight_beta': self.tf_weight_beta,
             'tf_weight_gamma': self.tf_weight_gamma,
             'tf_weight_delta': self.tf_weight_delta,
             'tf_weight_current': self.tf_weight_current,

             'tf_lowest_weight_a': self.tf_lowest_weight_a,
             'tf_lowest_weight_epsilon': self.tf_lowest_weight_epsilon,
             'tf_lowest_weight_beta': self.tf_lowest_weight_beta,
             'tf_lowest_weight_gamma' :self.tf_lowest_weight_gamma,
             'tf_lowest_weight_delta': self.tf_lowest_weight_delta,
             'tf_lowest_weight_current':  self.tf_lowest_weight_current,

             'tf_V': self.tf_V,
             'tf_v': self.tf_v,
             'tf_dVdt': self.tf_dVdt,
             'tf_dvdt': self.tf_dvdt,

             'training_loss': self.training_loss,
             'n_training_epoch': self.n_training_epoch,
             }
        return fhn_dl_model_instance

