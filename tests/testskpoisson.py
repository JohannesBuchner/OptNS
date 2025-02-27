import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import PoissonRegressor
from scipy.optimize import minimize

x = np.linspace(0, 10, 400)
A = 0 * x + 1
B = x
C = np.sin(x)**2
model = 30 * A + 5 * B + 50 * C

rng = np.random.RandomState(42)
data = rng.poisson(model)

plt.plot(x, model)
plt.scatter(x, data)
plt.savefig('testskpoisson.pdf')

X = np.transpose([A, B, C])
y = data
print(X.shape, y.shape)

reg = PoissonRegressor(solver='lbfgs', verbose=1, fit_intercept=False)
reg.fit(X, y)
print(reg.coef_, np.exp(reg.coef_))
print(reg.intercept_)
plt.plot(x, X @ reg.coef_)


print("--- manual approach: ---")
def minfunc(lognorms):
    lam = np.exp(lognorms) @ X.T
    loglike = data * np.log(lam) - lam
    # print('  ', lognorms, loglike.sum())
    return -loglike.sum()

x0 = np.log(np.median((data.reshape((-1, 1)) + 0.1) / (X + 0.1), axis=0))
print(x0, np.exp(x0))

t0 = time.time()
res = minimize(minfunc, x0, method='L-BFGS-B')
print(time.time() - t0, 'seconds passed')
print(res)
print(res.x, np.exp(res.x))
plt.plot(x, X @ np.exp(res.x))
plt.savefig('testskpoisson2.pdf')
