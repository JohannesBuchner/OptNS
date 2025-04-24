import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose
from optns.sampler import OptNS

def test_trivial_OLS():
    y = np.array([42.0])
    yerr = np.array([1.0])
    linear_param_names = ['A']
    nonlinear_param_names = []
    def compute_model_components(params):
        return np.transpose([[1.0], ])
    def nonlinear_param_transform(params):
        return params
    def linear_param_logprior(params):
        return 0
    
    np.random.seed(431)
    statmodel = OptNS(
        linear_param_names, nonlinear_param_names, compute_model_components,
        nonlinear_param_transform, linear_param_logprior,
        y, yerr**-2)

    # create a sampler from this
    optsampler = statmodel.ReactiveNestedSampler(
        log_dir='run-trivial-OLS', resume='overwrite')
    optresults = optsampler.run(max_num_improvement_loops=0, frac_remain=0.5)
    optsampler.print_results()
    #optsampler.plot()
    print(optresults['samples'])

    # this is a trivial case without free parameters:
    assert optresults['samples'].shape[1] == 0

    # get the full posterior:
    # this samples up to 1000 normalisations for each nonlinear posterior sample:
    fullsamples, weights, y_preds = statmodel.get_weighted_samples(optresults['samples'], 2500)
    assert_allclose(weights, 1. / len(weights))
    assert fullsamples.shape == (2500 * len(optresults['samples']), 1)

    # verify that samples are normal distributed
    assert_allclose(fullsamples.mean(), 42.0, rtol=0.001)
    assert_allclose(fullsamples.std(), 1.0, rtol=0.001)

    # to obtain equally weighted samples, we resample
    # this respects the effective sample size. If you get too few samples here,
    # crank up the number just above.
    samples, y_pred_samples = statmodel.resample(fullsamples, weights, y_preds)
    print(f'Obtained {len(samples)} equally weighted posterior samples')

    assert np.all(samples[:,0] == y_pred_samples[:,0])
    # verify that samples are normal distributed
    assert_allclose(samples.mean(), 42.0, rtol=0.001)
    assert_allclose(samples.std(), 1.0, rtol=0.001)

    # verify that samples are normal distributed
    assert_allclose(y_pred_samples.mean(), 42.0, rtol=0.001)
    assert_allclose(y_pred_samples.std(), 1.0, rtol=0.001)

def test_trivial_OLS_linearparam_priors():
    data_mean = 42.0
    prior_mean = 40.0
    prior_sigma = 3.45
    measurement_sigma = 0.312
    y = np.array([data_mean])
    yerr = np.array([measurement_sigma])
    
    # weighted sum of 42 +- 1 and 40 +- 1.0
    expected_mean = (data_mean * prior_sigma**2 + prior_mean * measurement_sigma**2) / (measurement_sigma**2 + prior_sigma**2)
    expected_std = ((measurement_sigma**-2 + prior_sigma**-2))**-0.5

    linear_param_names = ['A']
    nonlinear_param_names = []
    def compute_model_components(params):
        return np.transpose([[1.0], ])
    def nonlinear_param_transform(params):
        return params
    def linear_param_logprior(params):
        # 40 +- 2
        return -0.5 * ((params[:,0] - 40) / prior_sigma)**2
    
    np.random.seed(431)
    statmodel = OptNS(
        linear_param_names, nonlinear_param_names, compute_model_components,
        nonlinear_param_transform, linear_param_logprior,
        y, yerr**-2)

    # create a sampler from this
    optsampler = statmodel.ReactiveNestedSampler(
        log_dir='run-trivial-OLS', resume='overwrite')
    optresults = optsampler.run(max_num_improvement_loops=0, frac_remain=0.5)
    optsampler.print_results()
    fullsamples, weights, y_preds = statmodel.get_weighted_samples(optresults['samples'], 2500)
    assert not np.allclose(weights, 1. / len(weights)), 'expecting some reweighting!'
    assert fullsamples.shape == (2500 * len(optresults['samples']), 1)

    # verify that samples are normal distributed
    assert_allclose(fullsamples.mean(), data_mean, rtol=0.001)
    assert_allclose(fullsamples.std(), measurement_sigma, rtol=0.001)

    # to obtain equally weighted samples, we resample
    # this respects the effective sample size. If you get too few samples here,
    # crank up the number just above.
    samples, y_pred_samples = statmodel.resample(fullsamples, weights, y_preds)
    print(f'Obtained {len(samples)} equally weighted posterior samples')

    assert np.all(samples[:,0] == y_pred_samples[:,0])
    # verify that samples are normal distributed
    assert_allclose(samples.mean(), expected_mean, rtol=0.001)
    assert_allclose(samples.std(), expected_std, rtol=0.001)

    # verify that samples are normal distributed
    assert_allclose(y_pred_samples.mean(), expected_mean, rtol=0.001)
    assert_allclose(y_pred_samples.std(), expected_std, rtol=0.001)


SNR = 1.0
Ndata = 10

# first we generate some mock data:

x = np.linspace(0, 10, Ndata)
A = 0 * x + 1
B = x
C = exp(-x / 5)
noise = 1.0 + 0 * x
model_linear = 1 * A + 1 * B + 1 * C
model_nonlinear = 1 * A + 1 * C

rng = np.random.RandomState(42)
data_linear = rng.normal(model_linear, noise)
data_nonlinear = rng.normal(model_nonlinear, noise)

# set up function which computes the various model components:
# the parameters are:
nonlinear_param_names = ['tau']
def compute_model_components(params):
    tau, = params
    return np.transpose([np.exp(-x / tau)])

# set up a prior transform for these nonlinear parameters
def nonlinear_param_transform(cube):
    return cube * 10

linear_param_names = ['A', 'B']

# prior log-probability density function for the linear parameters:
def linear_param_logprior_loguniform(params):
    return -np.log(params[:,0]) + -np.log(params[:,1])
def linear_param_logprior_flat(params):
    return 0

"""

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
fullsamples, weights, y_preds = statmodel.get_weighted_samples(optresults['samples'][:400], 1000)
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


    
"""
