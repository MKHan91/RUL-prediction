from Bayesian_approach.Probability_distribution.define_distribution import *
from back_data.get_prior import *

t = [100, 101, 102]
t_pred = [0, 1, 2]
x = [10, 11, 12]

BETA = 25.0
ALPHA = 2.0

params = ParamDefine()

Phi_N = params.expand(np.array(t)[:, np.newaxis], params.identity_basis_function, bf_args=None)
Phi_test = params.expand(np.array(t_pred)[:, np.newaxis], params.identity_basis_function, bf_args=None)

S_0 = 1/ALPHA * np.eye(Phi_N.shape[1])

m_N, s_N = posterior_general(Phi=Phi_N, t=np.array(x)[:, np.newaxis], m_0=0, S_0=S_0, beta=BETA, zero_mean=True)

y_pred, y_uncertainty = posterior_predictive(Phi_test=Phi_test, m_N=m_N, S_N=s_N, beta=BETA)
y_pred_2, y_uncertainty_2 = posterior_predictive(Phi_test=Phi_N, m_N=m_N, S_N=s_N, beta=BETA)

y_pred = y_pred.ravel()
uncertainty = y_uncertainty.ravel()

plt.plot(t, x, '.')
# plt.plot(t_pred, y_pred[:len(t_pred)])
plt.show()