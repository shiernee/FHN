import sys
sys.path.insert(1, '/home/sawsn/Shiernee/FHN/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/PointCloud/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/FileIO/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/Utils/src/utils')
sys.path.insert(1, '/home/sawsn/Shiernee/Diffusion/src/utils')

from FileIO import FileIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    forward_folder = '../data/case2_2Dscatter_100/forward/'

    fileio = FileIO()
    fileio.assign_forward_folder(forward_folder)

    intp_coord_model_instances = fileio.read_physics_model_instance(0, 'intp_coord')

    intp_coord_axis1 = intp_coord_model_instances['intp_coord_axis1']
    intp_coord_axis2 = intp_coord_model_instances['intp_coord_axis2']
    coord = pd.DataFrame(intp_coord_model_instances['coord'])
    intp_u_axis1 = intp_coord_model_instances['intp_u_axis1']
    intp_u_axis2 = intp_coord_model_instances['intp_u_axis2']
    u = pd.DataFrame(intp_coord_model_instances['u'])

    coord.columns = ['x', 'y', 'z']

    coord_interest = coord.loc[(coord['x'] > 45) & (coord['x']< 55) & (coord['y'] > 45) & (coord['y']< 55)]

    for i in range(coord_interest.shape[0]):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        idx = coord_interest.index[i]

        coord_tmp = coord.loc[idx].values
        intp_coord_axis1_tmp = intp_coord_axis1[idx]
        intp_coord_axis2_tmp = intp_coord_axis2[idx]

        u_tmp = u.loc[idx].values
        intp_u_axis1_tmp = intp_u_axis1[idx]
        intp_u_axis2_tmp = intp_u_axis2[idx]

        ax.scatter(coord_tmp[0].reshape([-1, 1]), coord_tmp[1].reshape([-1, 1]), facecolor='None', edgecolors='r')
        ax.scatter(intp_coord_axis1_tmp[:, 0], intp_coord_axis1_tmp[:, 1], c=intp_u_axis1_tmp.squeeze())
        ax.scatter(intp_coord_axis2_tmp[:, 0], intp_coord_axis2_tmp[:, 1], c=intp_u_axis2_tmp.squeeze())
        ax.set_title('index : {}'.format(idx))
