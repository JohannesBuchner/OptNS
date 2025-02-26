import matplotlib.pyplot as plt
import numpy as np
import corner
from optns import OptNS

# first we generate some mock data:

x = np.linspace(0, 3, 40)
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

# lets now start using optimized nested sampling.

# set up function which computes the various model components:
# the parameters are:
nonlinear_param_names = ['f', 'phi', 'pow']
def compute_model_components(params):
    freq, phase, p = params
    periodic_signal = np.sin(2 * np.pi * x / freq + phase)
    periodic_power_signal = np.sign(periodic_signal) * np.abs(periodic_signal)**p
    return np.transpose([A, B, periodic_power_signal])

# set up a prior transform for these nonlinear parameters
def nonlinear_param_transform(cube):
    params = cube.copy()
    params[0] = 10**(cube[0] * 2 - 0.5)    # sine frequency
    params[1] = cube[1] * 2 * np.pi      # offset
    params[2] = cube[2] * 10      # power
    return params

# now for the linear (normalisation) parameters:
linear_param_names = ['a', 'b', 'c']
# set up a prior log-probability density function for these linear parameters:
def linear_param_logprior(params):
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

# create OptNS object, and give it all of these ingredients,
# as well as our data
statmodel = OptNS(
    linear_param_names, nonlinear_param_names, compute_model_components,
        nonlinear_param_transform, linear_param_logprior,
        data, noise**-2, positive=True)
# create a UltraNest sampler from this. You can pass additional arguments like here:
optsampler = statmodel.ReactiveNestedSampler(
    log_dir=f'gaussprofilefit{Ndata}-opt', resume=True)
# run the UltraNest optimized sampler on the nonlinear parameter space:
optresults = optsampler.run(max_num_improvement_loops=0, frac_remain=0.5)
optsampler.print_results()
optsampler.plot()

# now for postprocessing the results, we want to get the full posterior:
# this samples up to 1000 normalisations for each nonlinear posterior sample:
fullsamples, weights, y_preds = statmodel.get_weighted_samples(1000)
print(f'Obtained {len(fullsamples)} weighted posterior samples')

# to obtain equally weighted samples, we resample
# this respects the effective sample size. If you get too few samples here,
# crank up the number just above.
samples, y_pred_samples = statmodel.resample(fullsamples, weights, y_preds)
print(f'Obtained {len(samples)} equally weighted posterior samples')

# plot the fit
plt.figure(figsize=(15, 6))
plt.plot(x, model)
plt.errorbar(x, data, noise, capsize=2, elinewidth=0.5, linestyle=' ')
for y_pred in y_pred_samples[:40]:
    plt.plot(x, y_pred, ls='-', lw=1, alpha=0.1, color='k')
plt.savefig(f'gaussprofilefit{Ndata}-opt.pdf')
plt.close()

# make a corner plot:
mask = weights > 1e-6 * weights.max()
fig = corner.corner(
    fullsamples[mask,:], weights=weights[mask],
    labels=linear_param_names + nonlinear_param_names,
    show_titles=True, quiet=True,
    plot_datapoints=False, plot_density=False,
    levels=[0.9973, 0.9545, 0.6827, 0.3934], quantiles=[0.15866, 0.5, 0.8413],
    contour_kwargs=dict(linestyles=['-','-.',':','--'], colors=['navy','navy','navy','purple']),
    color='purple'
)
plt.savefig(f'gaussprofilefit{Ndata}-opt-corner.pdf')
plt.close()
