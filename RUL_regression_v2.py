from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from NonlinearRegression import Nonlinear_Regression
from glob import glob
import os
import sys
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='RUL regression model')
parser.add_argument('--type',            type=str,     help='the type of regression, linear, Root, Exponential',
                    default='linear')
parser.add_argument('--tc',              type=list,    help='current time',                 default=np.arange(15, 610, 40))
parser.add_argument('--thld_safe',       type=float,                                       default=1.45)
parser.add_argument('--thld_fault',      type=float,                                       default=2.2)
parser.add_argument('--thld_failure',    type=float,                                       default=9.0)
parser.add_argument('--range_pred',      type=int,                                          default=500)
parser.add_argument('--n_window',        type=int,    help='window for data fitting',         default=15)
parser.add_argument('--mv_size',         type=int,    help='moving average size',            default=30)
parser.add_argument('--degree',          type=int,    help='the degree of equation for linear regression', default=1)
parser.add_argument('--data_path',       type=str,    default='D:\\Onepredict_MK\\01 Solution 3 Team documents\\02 RUL Modeler\\data')
parser.add_argument('--save_path',       type=str,    default='D:\\Onepredict_MK\\01 Solution 3 Team documents\\02 RUL Modeler')
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

def current_time(*required):
    tc, n_window = required
    return np.arange(tc - n_window, tc)

def prediction_time(*required):
    tc, n_window, range_pred = required
    return np.arange(tc-args.n_window, tc+range_pred)

def step_time(*required):
    time, n_window = required

    return [time *20, time*20 + n_window]

# def plot_before_tc(tc, n_window, MA_data):
#     for time in np.arange(0, tc - n_window + 1): # 0 ~ 355
#         observed_data = MA_data[0:step_time(time, n_window)[1]]
#         if True in (observed_data >= args.thld_fault):
#             return observed_data, time
#         plt.title('RUL prediction - linear regression | time range: {}m ~ {}m'.format(step_time(time, n_window)[0], step_time(time, n_window)[1]))
#         plt.text(0, args.thld_safe + 0.05, str(args.thld_safe), color='c', fontsize=13, weight='bold')
#         plt.text(0, args.thld_fault + 0.05, str(args.thld_fault), color='r', fontsize=13, weight='bold')
#         plt.text(0, args.thld_failure + 0.05, str(args.thld_failure), color='r', fontsize=13, weight='bold')
#
#         plt.plot(MA_data, '.', label='total data')
#         plt.plot(np.arange(0, step_time(time, n_window)[1]), observed_data, '.', label='observed_data')
#
#         plt.axvline(x=step_time(time, n_window)[0])
#         plt.axvline(x=step_time(time, n_window)[1])
#         plt.axhline(y=args.thld_safe, color='c', linestyle='-', label='Threshold safe')
#         plt.axhline(y=args.thld_fault, color='r', linestyle='--', label='Threshold fault')
#         plt.axhline(y=args.thld_failure, color='r', linestyle='-', label='Threshold failure')
#
#         plt.axvspan(step_time(time, n_window)[0], step_time(time, n_window)[1], facecolor='gray', alpha=0.5)
#         plt.xlabel('Time (m)')
#         plt.ylabel('RMS')
#         plt.xlim([-10, 700])
#         plt.ylim([min(observed_data) - 0.2, 10])
#
#         plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.6))
#         plt.tight_layout()
#         plt.pause(0.5)
#         plt.clf()

def plot_gp(*required):
    mu, cov, X, X_train, Y_train, samples = required
    # samples = []
    # X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i + 1}')
    # plt.plot(X_train, Y_train, 'rx')

def main(rms_data):
    global model

    RUL_preds = []
    total_data = rms_data[0].flatten()     # (858, 1)

    fig = plt.figure(figsize=(19, 9))
    # ax1 = fig.add_subplot(2, 2, (1, 3))
    # ax2 = fig.add_subplot(2, 2, 4)

    "Moving Average"
    MA_data= np.convolve(total_data, np.ones((args.mv_size,)) / args.mv_size, mode='valid')[:610]   # (818, 1)
    RUL_true = [(num, rms) for num, rms in enumerate(MA_data) if 9.0 <= rms <= 9.1][0][0] # (818, 1)

    if args.type == 'linear':
        model = LinearRegression()
    elif args.type == 'exponential':
        model = Nonlinear_Regression()
    elif args.type == 'log':
        model = Nonlinear_Regression()
    elif args.type == 'lipow':
        model = Nonlinear_Regression()
    elif args.type == 'gom':
        model = Nonlinear_Regression()
    elif args.type == 'pow':
        model = Nonlinear_Regression()
    elif args.type == 'GP':
        kernel_RBF = kernels.ConstantKernel(1.0) * kernels.RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel=kernel_RBF, alpha=0.4**2)
    else:
        assert print('Unsupported regression model', file=sys.stderr)

    for cnt, tc in sorted(enumerate(args.tc)):
        ax1 = fig.add_subplot(2, 2, (1, 3))
        ax2 = fig.add_subplot(2, 2, 4)

        observed_data = MA_data[:tc]

        time_curr_range = current_time(tc, args.n_window)
        time_pred_range = prediction_time(tc, args.n_window, args.range_pred)

        if np.prod(observed_data[time_curr_range] >= args.thld_fault):
            if args.type == 'linear':
                model.fit(time_curr_range[:,np.newaxis], observed_data[time_curr_range])
                y_pred = model.predict(time_pred_range[:,np.newaxis])
                try:
                    RUL_pred = np.where(y_pred > args.thld_failure)[0][0] + (tc - args.n_window)
                    RUL_preds += [RUL_pred]
                    print('Predicted RUL: {}m'.format(RUL_pred), file=sys.stderr)
                except:
                    RUL_pred = 'Unestimatable'
                    RUL_preds += [RUL_preds[-1]]
                    print('Predicted RUL: {}'.format(RUL_pred), file=sys.stderr)

                ax1.plot(np.arange(tc-args.n_window,tc+args.range_pred), y_pred, '.')
                ax1.annotate('Predicted RUL: {}'.format(RUL_pred), xy=(tc, 4.8), fontsize=12, color='red', weight='bold')

            elif args.type == 'exponential':
                popt, _ = model.fit_exp(time_curr_range, observed_data[time_curr_range])
                y_pred = model.exponential_func(prediction_time(tc, args.n_window, args.range_pred), *popt)

                try:
                    RUL_pred = np.where(y_pred > args.thld_failure)[0][0] + (tc - args.n_window)
                    RUL_preds += [RUL_pred]
                    print('Predicted RUL: {}m'.format(RUL_pred), file=sys.stderr)
                except:
                    RUL_pred = 'Unestimatable'
                    RUL_preds += [RUL_preds[-1]]
                    print('Predicted RUL: {}'.format(RUL_pred), file=sys.stderr)

            elif args.type == 'log':
                popt, _ = model.fit_log(time_curr_range, observed_data[time_curr_range])
                y_pred = model.log_func(prediction_time(tc, args.n_window, args.range_pred), *popt)

                try:
                    RUL_pred = np.where(y_pred > args.thld_failure)[0][0] + (tc - args.n_window)
                    RUL_preds += [RUL_pred]
                    print('Predicted RUL: {}m'.format(RUL_pred), file=sys.stderr)
                except:
                    RUL_pred = 'Unestimatable'
                    RUL_preds += [RUL_preds[-1]]
                    print('Predicted RUL: {}'.format(RUL_pred), file=sys.stderr)

                ax1.plot(np.arange(tc-args.n_window, tc+args.range_pred), y_pred)
                ax1.annotate('Predicted RUL: {}'.format(RUL_pred), xy=(tc, 4.8), fontsize=12, color='red', weight='bold')

            elif args.type == 'lipow':
                popt, _ = model.fit_lipow(time_curr_range, observed_data[time_curr_range])
                y_pred = model.lipow_func(prediction_time(tc, args.n_window, args.range_pred), *popt)

                try:
                    RUL_pred = np.where(y_pred > args.thld_failure)[0][0] + (tc - args.n_window)
                    RUL_preds += [RUL_pred]
                    print('Predicted RUL: {}m'.format(RUL_pred), file=sys.stderr)
                except:
                    RUL_pred = 'Unestimatable'
                    RUL_preds += [RUL_preds[-1]]
                    print('Predicted RUL: {}'.format(RUL_pred), file=sys.stderr)

                ax1.plot(np.arange(tc - args.n_window, tc + args.range_pred), y_pred)
                ax1.annotate('Predicted RUL: {}'.format(RUL_pred), xy=(tc, 4.8), fontsize=12, color='red', weight='bold')

            elif args.type == 'gom':
                popt, _ = model.fit_gom(time_curr_range, observed_data[time_curr_range])
                y_pred = model.gom_func(prediction_time(tc, args.n_window, args.range_pred), *popt)

                try:
                    RUL_pred = np.where(y_pred > args.thld_failure)[0][0] + (tc - args.n_window)
                    RUL_preds += [RUL_pred]
                    print('Predicted RUL: {}m'.format(RUL_pred), file=sys.stderr)
                except:
                    RUL_pred = 'Unestimatable'
                    RUL_preds += [RUL_preds[-1]]
                    print('Predicted RUL: {}'.format(RUL_pred), file=sys.stderr)

                ax1.plot(np.arange(tc - args.n_window, tc + args.range_pred), y_pred)
                ax1.annotate('Predicted RUL: {}'.format(RUL_pred), xy=(tc, 4.8), fontsize=12, color='red', weight='bold')

            elif args.type == 'pow':
                popt, _ = model.fit_pow(time_curr_range, observed_data[time_curr_range])
                y_pred = model.pow_func(prediction_time(tc, args.n_window, args.range_pred), *popt)

                try:
                    RUL_pred = np.where(y_pred > args.thld_failure)[0][0] + (tc - args.n_window)
                    RUL_preds += [RUL_pred]
                    print('Predicted RUL: {}m'.format(RUL_pred), file=sys.stderr)
                except:
                    RUL_pred = 'Unestimatable'
                    RUL_preds += [RUL_preds[-1]]
                    print('Predicted RUL: {}'.format(RUL_pred), file=sys.stderr)

                ax1.plot(np.arange(tc - args.n_window, tc + args.range_pred), y_pred)
                ax1.annotate('Predicted RUL: {}'.format(RUL_pred), xy=(tc, 4.8), fontsize=12, color='red', weight='bold')

            elif args.type == 'GP':
                "MLE를 이용하여 데이터 fitting"
                model.fit(time_curr_range[:, np.newaxis], observed_data[time_curr_range])
                "Compute posterior predictive mean and variance"
                mu_s, cov_s = model.predict(time_pred_range[:, np.newaxis], return_cov=True)
                samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
                plot_gp(mu_s, cov_s, time_pred_range, time_curr_range[:, np.newaxis], observed_data, samples)
        else:
            RUL_preds += [590+args.tc[cnt]]

        ax1.set_title('RUL prediction - linear regression |  time range: {}m ~ {}m'.format(tc - args.n_window, tc))
        ax1.plot(MA_data, '.', label='total data')
        ax1.plot(observed_data, '.', label='observed_data')

        ax1.text(0, args.thld_safe + 0.05, str(args.thld_safe), color='c', fontsize=13, weight='bold')
        ax1.text(0, args.thld_fault + 0.05, str(args.thld_fault), color='r', fontsize=13, weight='bold')
        ax1.text(0, args.thld_failure + 0.05, str(args.thld_failure), color='r', fontsize=13, weight='bold')

        ax1.axvline(x=tc - args.n_window)
        ax1.axvline(x=tc)
        ax1.axhline(y=args.thld_safe, color='c', linestyle='-', label='Threshold safe')
        ax1.axhline(y=args.thld_fault, color='r', linestyle='--', label='Threshold fault')
        ax1.axhline(y=args.thld_failure, color='r', linestyle='-', label='Threshold failure')

        ax1.axvspan(tc - args.n_window, tc, facecolor='gray', alpha=0.5)

        ax1.set_xlabel('Times (m)')
        ax1.set_ylabel('RMS')
        ax1.set_xlim([-10, 900])
        ax1.set_ylim([min(observed_data) - 0.2, 10])
        ax1.annotate('True RUL: {}'.format(RUL_true), xy=(tc, 5), fontsize=12, weight='bold')

        ax2.set_title('RUL curve')
        ax2.set_xlabel('Times (m)')
        ax2.set_ylabel('RUL')
        ax2.set_xlim([-10, RUL_true+20])
        ax2.set_ylim([-10, RUL_true+20])
        ax2.plot([0, RUL_true], [RUL_true, 0], label='True RUL curve')
        ax2.plot(args.tc[:len(RUL_preds)], np.array(RUL_preds) - args.tc[:len(RUL_preds)], label='predicted RUL curve')

        ax1.legend(loc='upper left', bbox_to_anchor=(0.01, 0.6))
        ax2.legend(loc='upper left', bbox_to_anchor=(0.01, 0.6))

        plt.tight_layout()
        plt.pause(0.1)
        plt.clf()
    os.system('pause')
    # plt.show()

if __name__ == '__main__':
    rms_data = data_loader()
    main(rms_data)