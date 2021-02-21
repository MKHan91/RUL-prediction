from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from scipy.signal import medfilt
from sklearn.kernel_ridge import KernelRidge
from scipy.optimize import curve_fit
from glob import glob
import sys
import h5py
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='RUL regression model')
parser.add_argument('--mode',            type=str,    help='train mode according to data', default='train_v2')
parser.add_argument('--type',            type=str,    help='the type of regression, linear, Root, Exponential',       default='linear')
parser.add_argument('--tc',              type=list,    help='current time',                 default=[400, 420, 440, 460, 500, 520])
parser.add_argument('--thld_safe',         type=float,                                       default=1.45)
parser.add_argument('--thld_fault',        type=float,                                       default=2.2)
parser.add_argument('--thld_failure',      type=float,                                       default=8.0)
parser.add_argument('--range_pred',     type=int,                                          default=300)
parser.add_argument('--n_window',     type=int,    help='window for data cutting',         default=50)
parser.add_argument('--degree',          type=int,    help='the degree of equation for linear regression', default=1)
parser.add_argument('--data_path',       type=str,    default='D:\\Onepredict_MK\\01 Solution 3 Team documents\\02 RUL Modeler\\data')
args = parser.parse_args()

def data_loader():
    rms_data = []
    mat_file_list = sorted(glob(args.data_path + '/*.mat'))

    for num, mat_file in enumerate(mat_file_list):
        if not '_rms' in mat_file:
            continue
        f_01 = h5py.File(mat_file, 'r')
        for _, value in f_01.items():
            sflr_nf_01_arr = np.array(value)
            sflr_nf_01_arr_t = sflr_nf_01_arr.T

            rms_data += [sflr_nf_01_arr_t]

    return rms_data

def fault_lv_sampling(rms_data):
    true_target_val_data = np.zeros(shape=(100))

    target_val_data = rms_data[1][args.window_size:, :]

    idx = 0
    for target_element_val in target_val_data:
        if target_element_val >= args.fault_lv:
            true_target_val_data[idx] = target_element_val
            idx += 1
    index = np.where(true_target_val_data == 0)[0].tolist()
    true_target_val_data = np.delete(true_target_val_data, index)

    return true_target_val_data

def safe_lv_sampling(rms_data):
    true_target_tr_data = np.zeros(shape=(100))

    target_val_data = rms_data[1][:args.window_size, :]

    idx = 0
    for target_element_val in target_val_data:
        if target_element_val >= args.safe_lv:
            true_target_tr_data[idx] = target_element_val
            idx += 1
    index = np.where(true_target_tr_data == 0)[0].tolist()
    true_target_tr_data = np.delete(true_target_tr_data, index)

    return true_target_tr_data

def linear_regression_model(*required):
    "Get the coefficient of the equation"
    # input_tr, input_val, true_target_tr_data, sampled_true_tar_val_data = required
    input_tr, true_target_tr_data = required
    input_tr, true_target_tr_data = np.asarray(input_tr)[:, np.newaxis], np.asarray(true_target_tr_data)[:, np.newaxis]

    poly_features = PolynomialFeatures(degree=args.degree)
    coefficients = poly_features.fit_transform(input_tr, true_target_tr_data)
    # extra_poly = poly_features.fit_transform(input_val, sampled_true_tar_val_data)
    # extra_poly = poly_features.fit_transform(input_tr, np.hstack((np.asarray(arr), target_train_data[:, 0])))

    model = LinearRegression()
    model.fit(coefficients[:true_target_tr_data.shape[0], :], true_target_tr_data)
    # model.fit(coefficients, true_target_tr_data[:coefficients.shape[0], :])
    "Predicting polynomial equation"
    y_train_pred = model.predict(coefficients)
    # y_train_pred = model.predict(coefficients[:true_target_tr_data.shape[0], :])
    # y_train_extra = model.predict(extra_poly)

    # return y_train_pred, y_train_extra
    return y_train_pred

def main(rms_data):
    if args.mode == 'train':
        # rnd_tmp = random.uniform(2.0, 4.0)
        target_train_data = rms_data[1][:args.window_size, :]
        sampled_tar_tr_time = np.where(target_train_data >= args.safe_lv)[0]

        true_target_tr_data = safe_lv_sampling(rms_data)    # 12
        true_target_val_data = fault_lv_sampling(rms_data)  # 67

        input_tr = sampled_tar_tr_time
        input_tr = input_tr[:, np.newaxis]

        "Validation data sampling"
        for val_window_size in args.val_window_size:
            sampled_true_tar_val_data = [true_target_val_data[i] for i in range(len(true_target_val_data))]
            sampled_true_tar_val_data = sampled_true_tar_val_data[:val_window_size]

            input_val = np.asarray(range(target_train_data.shape[0], target_train_data.shape[0]+len(sampled_true_tar_val_data))) # 900 ~ 910
            input_val = input_val[:, np.newaxis]

            y_train_pred, y_train_extra = linear_regression_model(input_tr, input_val, true_target_tr_data, sampled_true_tar_val_data)

            plt.title('RMS Regression - degree:{} / val data num: {}/{}'.format(args.degree, val_window_size, len(true_target_val_data)))
            plt.plot(target_train_data, '.', label='training data')
            plt.plot(np.arange(len(sampled_true_tar_val_data))+args.window_size, sampled_true_tar_val_data, '.', label='validation data')
            plt.plot(input_tr, y_train_pred, color='m', label='Polynomial equation')
            plt.plot(np.arange(len(sampled_true_tar_val_data))+args.window_size, y_train_extra, color='y', label='Extra-polated equation')
            "Fault level plot"
            plt.axhline(y=args.fault_lv, color='r', linestyle='-', label='Fault level')
            "Safe level plot"
            plt.axhline(y=args.safe_lv, color='c', linestyle='-', label='Safe level')

            plt.legend(loc='best')
            plt.xlabel('Time')
            plt.ylabel('RMS')
            plt.show()

    elif args.mode == 'train_v2':
        # Step 1: data preparation
        N = 41
        total_data = rms_data[0].flatten()     # (858, 1)
        "Moving Average"
        MA_data= np.convolve(total_data, np.ones((N,)) / N, mode='valid')[:600]   # (818, 1)

        for tc in sorted(args.tc):
            observed_data = MA_data[:tc]

            model = LinearRegression()

            if np.prod(observed_data[np.arange(tc-args.n_window, tc)] >= args.thld_fault):
                model.fit(np.arange(tc-args.n_window, tc)[:,np.newaxis], observed_data[np.arange(tc-args.n_window, tc)])
                y_pred = model.predict(np.arange(tc-args.n_window, tc+args.range_pred)[:,np.newaxis])

                RUL_pred = np.where(y_pred > args.thld_failure)[0][0] + tc
                sys.stderr.write('Predicted RUL: {}m |'.format(RUL_pred))


                plt.title('RUL prediction - linear regression | current time: {}m'.format(tc))
                plt.plot(MA_data, '.', label='total data')
                plt.plot(observed_data, '.', label='observed_data')
                plt.plot(np.arange(tc-args.n_window,tc+args.range_pred), y_pred, '.')

                plt.axvline(x=tc-args.n_window)
                plt.axvline(x=tc, label='Current time')
                plt.axhline(y=args.thld_fault, color='r', linestyle='--', label='Threshold fault')
                plt.axhline(y=args.thld_safe, color='c', linestyle='-', label='Threshold safe')
                plt.axhline(y=args.thld_failure, color='r', linestyle='-', label='Threshold failure')

                plt.axvspan(tc-args.n_window, tc, facecolor='gray', alpha=0.5)
                plt.xlabel('Time (m)')
                plt.ylabel('RMS')
                plt.xlim([-10, 800])
                plt.ylim([min(observed_data)-0.2, 10])

                plt.legend(loc='best')
                plt.show()

if __name__ == '__main__':
    rms_data = data_loader()
    if args.type == 'linear':
        main(rms_data)
