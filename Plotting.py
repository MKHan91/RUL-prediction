from glob import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

def with_plot():
    mat_file_list = sorted(glob(data_path+'/*.mat'))

    for num, mat_file in enumerate(mat_file_list):
        if not '_rms' in mat_file:
            continue
        name = mat_file.split('/')[-1].split('.')[0]
        f_01 = h5py.File(mat_file, 'r')

        for _, value in f_01.items():
            sflr_nf_01_arr = np.array(value)
            sflr_nf_01_arr_t = sflr_nf_01_arr.T

            plt.plot(sflr_nf_01_arr_t, '.', label='{}'.format(name))
            plt.legend(loc='best')
            plt.show()


if __name__ == '__main__':
    data_path = '/home/onepredict/Myungkyu/RUL_modeler/data'