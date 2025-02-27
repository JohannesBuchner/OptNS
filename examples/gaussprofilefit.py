import matplotlib.pyplot as plt
import numpy as np
import tqdm
import corner
from ultranest import ReactiveNestedSampler

x = np.linspace(0, 10, 40)
A = 0 * x + 1
B = x
C = np.sin(x + 2)**2
model = 3 * A + 0.5 * B + 5 * C
noise = 0.5 + 0.1 * x

rng = np.random.RandomState(42)
data = rng.normal(model, noise)
Ndata = len(data)
plt.figure(figsize=(15, 6))
plt.plot(x, model)
plt.errorbar(x, data, noise, capsize=2, elinewidth=0.5, linestyle=' ')
plt.savefig(f'gaussprofilefit{Ndata}-data.pdf')
plt.close()

def transform(cube):
    params = cube.copy()
    params[0] = 10**(cube[0] * 10 - 5)   # constant background
    params[1] = cube[1] * 10             # incline strength
    params[2] = 10**(cube[2] * 10 - 5)   # sine amplitude
    params[3] = 10**(cube[3] * 2 - 0.5)    # sine frequency
    params[4] = cube[4] * np.pi      # offset
    params[5] = cube[5] * 10             # power
    return params

def compute_model_components(params):
    freq, phase, p = params
    periodic_signal = np.sin(2 * np.pi * x / freq + phase)
    periodic_power_signal = np.abs(periodic_signal)**p
    return np.transpose([A, B, periodic_power_signal])

def loglikelihood_simple(params):
    a, b, c, freq, phase, p = params
    periodic_signal = np.sin(2 * np.pi * x / freq + phase)
    periodic_power_signal = np.abs(periodic_signal)**p
    y_pred = a * A + b * B + c * periodic_power_signal
    return -0.5 * np.sum(((y_pred - data) / noise)**2)

def loglikelihood(params):
    a, b, c, freq, phase, p = params
    y_pred = params[:3] @ compute_model_components(params[3:]).T
    return -0.5 * np.sum(((y_pred - data) / noise)**2)

from optns.profilelike import ComponentModel

statmodel = ComponentModel(3, data, noise**-2)

def opttransform(cube):
    params = cube.copy()
    params[0] = 10**(cube[0] * 2 - 0.5)    # sine frequency
    params[1] = cube[1] * np.pi      # offset
    params[2] = cube[2] * 10      # power
    return params

def opt_linearparam_logprior(params):
    # a log-uniform prior, a uniform prior, and a log-uniform prior
    logp = -np.log(params[:,0])
    logp += -np.log(params[:,2])
    # limits:
    m1 = np.logical_and(params[:,0] > 1e-5, params[:,0] < 1e5)
    m2 = np.logical_and(params[:,1] > 0, params[:,1] < 10)
    m3 = np.logical_and(params[:,2] > 1e-5, params[:,2] < 1e5)
    logp[~m1] = -np.inf
    logp[~m2] = -np.inf
    logp[~m3] = -np.inf
    return logp



def optloglikelihood(nonlinear_params):
    return statmodel.loglike_gauss(compute_model_components(nonlinear_params))

def optlinearsample(nonlinear_params, size):
    X = compute_model_components(nonlinear_params)
    linear_params, loglike_proposal, loglike_target = statmodel.sample_gauss(X, size)
    Nsamples, Nlinear = linear_params.shape
    y_pred = linear_params @ X.T
    # logl_full = -0.5 * np.sum(((y_pred - data) / noise)**2, axis=1)
    logprior = opt_linearparam_logprior(linear_params)
    params = np.empty((Nsamples, len(nonlinear_params) + Nlinear))
    params[:, :Nlinear] = linear_params
    params[:, Nlinear:] = nonlinear_params.reshape((1, -1))
    return y_pred, params, loglike_target + logprior - loglike_proposal - np.log(Nsamples)

linear_param_names = ['a', 'b', 'c']
nonlinear_param_names = ['f', 'phi', 'pow']
param_names = linear_param_names + nonlinear_param_names

sampler = ReactiveNestedSampler(
    param_names, loglikelihood, transform=transform,
    log_dir=f'gaussprofilefit{Ndata}', resume=True)
results = sampler.run(max_num_improvement_loops=0, frac_remain=0.5)
sampler.print_results()
sampler.plot()

plt.figure(figsize=(15, 6))
plt.plot(x, model)
plt.errorbar(x, data, noise, capsize=2, elinewidth=0.5, linestyle=' ')
for params in results['samples'][:40]:
    a, b, c, freq, phase, p = params
    y_pred = params[:3] @ compute_model_components(params[3:]).T
    plt.plot(x, y_pred, ls='-', lw=1, alpha=0.1, color='k')
plt.savefig(f'gaussprofilefit{Ndata}.pdf')
plt.close()


optsampler = ReactiveNestedSampler(
    nonlinear_param_names, optloglikelihood, transform=opttransform,
    log_dir=f'gaussprofilefit{Ndata}-opt', resume=True)
optresults = optsampler.run(max_num_improvement_loops=0, frac_remain=0.5)
optsampler.print_results()
optsampler.plot()

# go through posterior samples and sample normalisations
size = 1000
Nsampled = 0
logweights = np.empty(len(optresults['samples']) * size)
y_preds = np.empty((len(optresults['samples']) * size, len(data)))
fullsamples = np.empty((len(optresults['samples']) * size, len(param_names)))
for i, nonlinear_params in enumerate(tqdm.tqdm(optresults['samples'])):
    y_pred_i, fullsamples_i, logweights_i = optlinearsample(
        nonlinear_params, size=size)
    Nsampled_i = len(logweights_i)
    jlo = Nsampled
    jhi = Nsampled + Nsampled_i
    y_preds[jlo : jhi, :] = y_pred_i
    fullsamples[jlo : jhi, :] = fullsamples_i
    logweights[jlo : jhi] = logweights_i
    Nsampled += Nsampled_i 

print(f'Obtained {Nsampled} weighted posterior samples')
y_preds = y_preds[:Nsampled,:]
fullsamples = fullsamples[:Nsampled,:]
logweights = logweights[:Nsampled]


weights = np.exp(logweights - logweights.max())
weights /= weights.sum()
rejection_sampled_indices = rng.choice(len(weights), p=weights, size=40000)

plt.figure(figsize=(15, 6))
plt.plot(x, model)
plt.errorbar(x, data, noise, capsize=2, elinewidth=0.5, linestyle=' ')
for i in tqdm.tqdm(rejection_sampled_indices[:40]):
    y_pred = y_preds[i]
    plt.plot(x, y_pred, ls='-', lw=1, alpha=0.1, color='k')
plt.savefig(f'gaussprofilefit{Ndata}-opt.pdf')
plt.close()

mask = weights > 1e-6 * weights.max()
fig = corner.corner(
    fullsamples[mask,:], weights=weights[mask],
    labels=param_names, show_titles=True, quiet=True,
    plot_datapoints=False, plot_density=False,
    levels=[0.9973, 0.9545, 0.6827, 0.3934], quantiles=[0.15866, 0.5, 0.8413],
    contour_kwargs=dict(linestyles=['-','-.',':','--'], colors=['navy','navy','navy','purple']),
    color='purple'
)
plt.savefig(f'gaussprofilefit{Ndata}-opt-corner.pdf')
plt.close()
