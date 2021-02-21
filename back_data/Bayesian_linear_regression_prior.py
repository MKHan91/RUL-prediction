import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import matplotlib.pyplot as plt
from collections import namedtuple
from Bayesian_approach.Probability_distribution.define_distribution import *
from Bayesian_approach.Probability_distribution.plot_distribution import *
from back_data.get_prior import *

def main():
    util = utilityFunc()
    dist = Distribution(args.thld_safe, args.thld_fault, args.thld_failure)

    RUL_error, RUL_preds = [], []
    mean_coeff_back, cov_coeff_back, tc_range = 0, 0, 0

    rms_data = util.data_loader()
    RUL_true = [(num, rms) for num, rms in enumerate(rms_data) if args.thld_failure <= rms][0][0]  # (818, 1)

    if args.data_type == 'mat':
        mean_coeff_back, cov_coeff_back = back_data_coeff_mat(rms_data)
        tc_range = np.arange(args.n_window, 615, 5)

    elif args.data_type == 'csv':
        mean_coeff_back, cov_coeff_back = back_data_coeff_csv(rms_data)
        tc_range = np.arange(args.n_window, len(rms_data)+5, 5)

    for cnt, tc in enumerate(tc_range):
        ax_data = fig.add_subplot(4, 5, (1, 18))
        ax_rul = fig.add_subplot(4, 5, (14, 20))
        ax_error = fig.add_subplot(4, 5, (4, 10))

        observed_data = rms_data[:tc]

        time_curr_range = util.current_time(tc, args.n_window)
        time_pred_range = util.prediction_time(tc, args.n_window, args.range_pred)

        if min(observed_data[time_curr_range]) >= args.thld_fault:
            Phi_N = util.expand(time_curr_range[:, np.newaxis], util.identity_basis_function, bf_args=None)
            Phi_test = util.expand(time_pred_range[:, np.newaxis], util.identity_basis_function, bf_args=None)

            # S_0 = 1/ALPHA*np.eye(Phi_N.shape[1])
            # beta = 1 / np.var(observed_data[time_curr_range])
            # cov_coeff_back[0][0] = cov_coeff_back[0][0] * 1e4
            # cov_coeff_back[1][0] = cov_coeff_back[1][0] * 0
            m_N, s_N = posterior_general(Phi=Phi_N, t=observed_data[time_curr_range][:, np.newaxis], m_0=mean_coeff_back,
                                         S_0=cov_coeff_back, beta=BETA)

            y_pred, y_uncertainty = posterior_predictive(Phi_test=Phi_test, Phi_N=Phi_N, m_N=m_N, S_N=s_N, beta=BETA)
            y_pred = y_pred.ravel()
            uncertainty = y_uncertainty.ravel()

            try:
                RUL_pred = np.where(y_pred > args.thld_failure)[0][0] + (tc - args.n_window)
                RUL_preds.append(RUL_pred)
                "Calculate RUL error"
                RUL_error = util.RUL_error(RUL_preds, RUL_true, RUL_error, tc_range)
            except:
                RUL_pred = 'Unpredictable'
                RUL_preds.append(RUL_preds[-1])

            ax_data.annotate('Predicted RUL: {}'.format(RUL_pred), xy=(tc, args.pos - 0.25), fontsize=12, weight='bold', color='r')
            ax_data.plot(time_pred_range, y_pred, c='k', label='Prediction_y')
            ax_data.fill_between(time_pred_range.ravel(), y_pred + uncertainty, y_pred - uncertainty, alpha=0.2)

            dist.plot_RUL_error(ax_error, RUL_preds[:len(RUL_error)], RUL_error, tc_range, args)
            # os.system('pause')
        else:
            RUL_preds.append(RUL_true + tc_range[cnt])

        dist.plot_data(ax_data, rms_data, observed_data, RUL_true, tc, args)
        dist.plot_RUL_curve(ax_rul, RUL_preds, RUL_true, tc_range, args)

        if args.capture:
            path_capture = os.path.join('\\'.join(args.data_path.split('\\')[:-1]), 'about_document'
                                        ,'Regression capture image', 'Bayesian linear regression')
            if tc == 280:
                plt.savefig(path_capture+'\\{}_{}time'.format('BLR based on back-data', tc))

        plt.tight_layout()
        plt.pause(0.05)
        plt.clf()
    os.system('pause')

if __name__ == '__main__':
    BETA = 500.0
    # BETA = 2.0
    ALPHA = 2.0
    main()