from sklearn.linear_model import LinearRegression
from NonlinearRegression import Nonlinear_Regression
from glob import glob
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

def data_loader(data_path):
    rms_data = []
    mat_file_list = sorted(glob(data_path + '\\*.mat'))
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
    t = tc - n_window
    return np.arange(t, tc)

def prediction_time(*required):
    tc, n_window, range_pred = required
    return np.arange(tc-n_window, tc+range_pred)

def main(rms_data):
    global model
    RUL_preds = []

    total_data = rms_data[0].flatten()     # (858, 1)

    fig = plt.figure(figsize=(19, 9))

    "Moving Average"
    MA_data= np.convolve(total_data, np.ones((mv_size,)) / mv_size, mode='valid')[:610]   # (818, 1)
    RUL_true = [(num, rms) for num, rms in enumerate(MA_data) if thld_failure <= rms][0][0] # (818, 1)

    if type_ == 'linear':
        model = LinearRegression()
    elif type_ == 'exponential':
        model = Nonlinear_Regression()
    elif type_ == 'log':
        model = Nonlinear_Regression()
    elif type_ == 'Lloyd-Lipow':
        model = Nonlinear_Regression()
    elif type_ == 'gompertz':
        model = Nonlinear_Regression()
    elif type_ == 'power':
        model = Nonlinear_Regression()
    else:
        assert print('Unsupported regression model', file=sys.stderr)

    for cnt, tc in sorted(enumerate(tc_range)):
        ax1 = fig.add_subplot(4, 5, (1, 18))
        ax2 = fig.add_subplot(4, 5, (14, 20))

        observed_data = MA_data[:tc]

        time_curr_range = current_time(tc, n_window)
        time_pred_range = prediction_time(tc, n_window, range_pred)

        ax1.set_title('RUL prediction - {} regression |  time range: {}m ~ {}m'.format(type_, tc - n_window, tc))
        ax1.plot(MA_data, '.', label='total data')
        ax1.plot(observed_data, '.', label='observed_data')

        ax1.text(0, thld_safe + 0.05, str(thld_safe), color='c', fontsize=13, weight='bold')
        ax1.text(0, thld_fault + 0.05, str(thld_fault), color='r', fontsize=13, weight='bold')
        ax1.text(0, thld_failure + 0.05, str(thld_failure), color='r', fontsize=13, weight='bold')

        ax1.axvline(x=tc - n_window)
        ax1.axvline(x=tc)
        ax1.axhline(y=thld_safe, color='c', linestyle='-', label='Threshold safe')
        ax1.axhline(y=thld_fault, color='r', linestyle='--', label='Threshold fault')
        ax1.axhline(y=thld_failure, color='r', linestyle='-', label='Threshold failure')

        ax1.axvspan(tc - n_window, tc, facecolor='gray', alpha=0.5)

        ax1.set_xlabel('Times (m)')
        ax1.set_ylabel('RMS')
        ax1.set_xlim([-10, 900])
        ax1.set_ylim([min(observed_data) - 0.2, 10])
        ax1.annotate('True RUL: {}'.format(RUL_true), xy=(tc, 5), fontsize=12, weight='bold')

        if np.prod(observed_data[time_curr_range] >= thld_fault):
            if type_ == 'linear':
                model.fit(time_curr_range[:,np.newaxis], observed_data[time_curr_range])
                y_pred = model.predict(time_pred_range[:,np.newaxis])
                try:
                    RUL_pred = np.where(y_pred > thld_failure)[0][0] + (tc - n_window)
                    RUL_preds += [RUL_pred]
                    print('Predicted RUL: {}m'.format(RUL_pred), file=sys.stderr)
                except:
                    RUL_pred = 'Unestimatable'
                    RUL_preds += [RUL_preds[-1]]
                    print('Predicted RUL: {}'.format(RUL_pred), file=sys.stderr)

                ax1.plot(np.arange(tc-n_window,tc+range_pred), y_pred, 'g-', linewidth=3)
                ax1.annotate('Predicted RUL: {}'.format(RUL_pred), xy=(tc, 4.8), fontsize=12, color='red', weight='bold')

            elif type_ == 'exponential':
                popt, _ = model.fit_exp(time_curr_range, observed_data[time_curr_range])
                y_pred = model.exponential_func(prediction_time(tc, n_window, range_pred), *popt)

                try:
                    RUL_pred = np.where(y_pred > thld_failure)[0][0] + (tc - n_window)
                    RUL_preds += [RUL_pred]
                    print('Predicted RUL: {}m'.format(RUL_pred), file=sys.stderr)
                except:
                    RUL_pred = 'Unestimatable'
                    RUL_preds += [RUL_preds[-1]]
                    print('Predicted RUL: {}'.format(RUL_pred), file=sys.stderr)

                ax1.plot(np.arange(tc-n_window,tc+range_pred)[:len(y_pred)], y_pred, 'g-', linewidth=3)
                ax1.annotate('Predicted RUL: {}'.format(RUL_pred), xy=(tc, 4.8), fontsize=12, color='red', weight='bold')

            elif type_ == 'log':
                popt, _ = model.fit_log(time_curr_range, observed_data[time_curr_range])
                y_pred = model.log_func(prediction_time(tc, n_window, range_pred), *popt)

                try:
                    RUL_pred = np.where(y_pred > thld_failure)[0][0] + (tc - n_window)
                    RUL_preds += [RUL_pred]
                    print('Predicted RUL: {}m'.format(RUL_pred), file=sys.stderr)
                except:
                    RUL_pred = 'Unestimatable'
                    RUL_preds += [RUL_preds[-1]]
                    print('Predicted RUL: {}'.format(RUL_pred), file=sys.stderr)

                ax1.plot(np.arange(tc-n_window, tc+range_pred), y_pred, 'g-', linewidth=3)
                ax1.annotate('Predicted RUL: {}'.format(RUL_pred), xy=(tc, 4.8), fontsize=12, color='red', weight='bold')
            elif type_ == 'Lloyd-Lipow':
                popt, _ = model.fit_lipow(time_curr_range, observed_data[time_curr_range])
                y_pred = model.lipow_func(prediction_time(tc, n_window, range_pred), *popt)

                try:
                    RUL_pred = np.where(y_pred > thld_failure)[0][0] + (tc - n_window)
                    RUL_preds += [RUL_pred]
                    print('Predicted RUL: {}m'.format(RUL_pred), file=sys.stderr)
                except:
                    RUL_pred = 'Unestimatable'
                    RUL_preds += [RUL_preds[-1]]
                    print('Predicted RUL: {}'.format(RUL_pred), file=sys.stderr)

                ax1.plot(np.arange(tc - n_window, tc + range_pred), y_pred, 'g-', linewidth=3)
                ax1.annotate('Predicted RUL: {}'.format(RUL_pred), xy=(tc, 4.8), fontsize=12, color='red',
                             weight='bold')

            elif type_ == 'gompertz':
                popt, _ = model.fit_gom(time_curr_range, observed_data[time_curr_range])
                y_pred = model.gom_func(prediction_time(tc, n_window, range_pred), *popt)

                try:
                    RUL_pred = np.where(y_pred > thld_failure)[0][0] + (tc - n_window)
                    RUL_preds += [RUL_pred]
                    print('Predicted RUL: {}m'.format(RUL_pred), file=sys.stderr)
                except:
                    RUL_pred = 'Unestimatable'
                    RUL_preds += [RUL_preds[-1]]
                    print('Predicted RUL: {}'.format(RUL_pred), file=sys.stderr)

                ax1.plot(np.arange(tc - n_window, tc + range_pred), y_pred, 'g-', linewidth=3)
                ax1.annotate('Predicted RUL: {}'.format(RUL_pred), xy=(tc, 4.8), fontsize=12, color='red',
                             weight='bold')

            elif type_ == 'power':
                popt, _ = model.fit_pow(time_curr_range, observed_data[time_curr_range])
                y_pred = model.pow_func(prediction_time(tc, n_window, range_pred), *popt)

                try:
                    RUL_pred = np.where(y_pred > thld_failure)[0][0] + (tc - n_window)
                    RUL_preds += [RUL_pred]
                    print('Predicted RUL: {}m'.format(RUL_pred), file=sys.stderr)
                except:
                    RUL_pred = 'Unestimatable'
                    RUL_preds += [RUL_preds[-1]]
                    print('Predicted RUL: {}'.format(RUL_pred), file=sys.stderr)

                ax1.plot(np.arange(tc - n_window, tc + range_pred), y_pred, 'g-', linewidth=3)
                ax1.annotate('Predicted RUL: {}'.format(RUL_pred), xy=(tc, 4.8), fontsize=12, color='red',
                             weight='bold')
        else:
            RUL_preds += [590 + tc_range[cnt]]

        ax2.set_title('RUL curve')
        ax2.set_xlabel('Times (m)')
        ax2.set_ylabel('RUL')
        ax2.set_xlim([-10, RUL_true+20])
        ax2.set_ylim([-10, RUL_true+20])
        ax2.plot([0, RUL_true], [RUL_true, 0], label='True RUL curve')
        ax2.plot(tc_range[:len(RUL_preds)], np.array(RUL_preds) - tc_range[:len(RUL_preds)], label='predicted RUL curve')

        ax1.legend(loc='upper left', bbox_to_anchor=(0.01, 0.6))
        ax2.legend(loc='upper left', bbox_to_anchor=(0.01, 0.6))

        plt.tight_layout()
        # "Screen capture at specific time "
        # if tc == 605:
        #     plt.savefig('D:\\Onepredict_MK\\01 Solution 3 Team documents\\02 RUL Modeler\\관련 자료\\Regression capture image\\{} regression_{}time'.format(type_, tc))
        #     print('Capture done!')
        plt.pause(speed_search)
        plt.clf()
    os.system('pause')

def input_func():
    data_path = input('데이터 주소를 입력하세요: ')
    if data_path == '':
        data_path = os.path.dirname(os.path.abspath(__file__)) + '\\data'

    mv_size = input('moving average 크기를 입력해 주세요 [Default: 45]: ')
    if mv_size == '':
        mv_size = 45

    type_ = input('모델을 선정하세요 [Default: linear] (사용 가능한 모델:linear, exponential, power, log, gompertz, Lloyd-Lipow): ')
    if type_ == '':
        type_ = 'linear'

    t_interval = input("시간 간격을 입력하세요 [Default: 5]: ")
    n_window = input('윈도우의 크기를 정하세요 [Default: 15]: ')
    if t_interval == '':
        t_interval = 5
    if n_window == '':
        n_window = 15
    tc_range = np.arange(int(n_window), 610, int(t_interval))

    speed_search = input('탐색 속도를 정하세요 [Default: 0.05]: ')
    if speed_search == '':
        speed_search = 0.05

    range_pred = input('예측 범위를 정해주세요: ')
    if range_pred == '':
        range_pred = 500

    thld_safe = input('Safe Threshold를 정해주세요 [Default: 1.45]: ')
    if thld_safe == '':
        thld_safe = 1.45

    thld_fault = input('Fault Threshold를 정해주세요 [Default: 2.2]: ')
    if thld_fault == '':
        thld_fault = 2.2

    thld_failure = input('Failure Threshold를 정해주세요 [Default: 9.0]: ')
    if thld_failure == '':
        thld_failure = 9.0
    return data_path, int(mv_size), type_, tc_range, float(speed_search), int(n_window), int(range_pred), float(thld_safe), float(thld_fault), float(thld_failure)

if __name__ == '__main__':
    data_path, mv_size, type_, tc_range, speed_search, n_window, range_pred, thld_safe, thld_fault, thld_failure = input_func()
    rms_data = data_loader(data_path)
    main(rms_data)