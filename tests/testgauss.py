import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

x = np.linspace(0, 10, 400)
A = 0 * x + 1
B = x
C = np.sin(x)**2
model = 3 * A + 0.5 * B + 5 * C
noise = 0.5 + 0.1 * x

rng = np.random.RandomState(42)
data = rng.normal(model, noise)

plt.figure(figsize=(15, 6))
plt.plot(x, model)
plt.errorbar(x, data, noise, capsize=2, elinewidth=0.5, linestyle=' ')
plt.savefig('testgauss.pdf')

X = np.transpose([A, B, C])
y = data
# inverse variance
sample_weight = noise**-2
print(X.shape, y.shape)

reg = LinearRegression(positive=True, fit_intercept=False)
reg.fit(X, y, sample_weight)
print(reg.coef_)
print(reg.intercept_)
print(dir(reg))
plt.plot(x, X @ reg.coef_, ls='--')

print("--- manual approach: ---")
def minfunc(lognorms):
    lam = np.exp(lognorms) @ X.T
    loglike = (data - lam)**2
    # print('  ', lognorms, loglike.sum())
    return -loglike.sum()

x0 = np.log(np.nanmedian(np.clip(data.reshape((-1, 1)) / X, 1e-6, None), axis=0))
print(x0, np.exp(x0))

t0 = time.time()
res = minimize(minfunc, x0, method='L-BFGS-B')
print(time.time() - t0, 'seconds passed')
print(res)
print(res.x, np.exp(res.x))
plt.plot(x, X @ np.exp(res.x))
plt.savefig('testgauss2.pdf')
