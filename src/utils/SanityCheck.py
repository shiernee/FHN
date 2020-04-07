import numpy as np

class SanityCheck:
    def __init__(self):
        return

    def check_if_equal_coordA_coordB(self, coordA, coordB):
        assert np.ndim(coordA) == 2, 'coordinates dimension should be 2D array'
        assert np.ndim(coordB) == 2, 'coordinates dimension should be 2D array'
        assert coordA.shape[-1] == 3, 'coordinate shape should be an (n_coord x 3)'
        assert coordB.shape[-1] == 3, 'coordinate shape should be an (n_coord x 3)'

        if np.sum(coordB - coordA) > 1e-5:
            raise Exception('coordA does not match coordB')
        print('coordA and coordB MATCHED')
        return
