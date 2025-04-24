import numpy as np
from numpy import exp
from numpy.testing import assert_allclose
from optns.sampler import OptNS
import scipy.stats
from sklearn.linear_model import LinearRegression
from optns.profilelike import GaussModel
from ultranest import ReactiveNestedSampler
import matplotlib.pyplot as plt
import corner
import joblib


ultranest_run_kwargs = dict(max_num_improvement_loops=0, show_status=False, viz_callback=None)


y0 = np.array([42.0])
yerr0 = np.array([1.0])
linear_param_names0 = ['A']
nonlinear_param_names0 = []
def compute_model_components0(params):
    return np.transpose([[1.0], ])
def nonlinear_param_transform0(params):
    return params
def linear_param_logprior0(params):
    return 0

def test_trivial_OLS():
    np.random.seed(431)
    statmodel = OptNS(
        linear_param_names0, nonlinear_param_names0, compute_model_components0,
        nonlinear_param_transform0, linear_param_logprior0,
        y0, yerr0**-2)

    # create a sampler from this
    optsampler = statmodel.ReactiveNestedSampler()
    optresults = optsampler.run(**ultranest_run_kwargs)

    # this is a trivial case without free parameters:
    assert optresults['samples'].shape[1] == 0

    # get the full posterior:
    # this samples up to 1000 normalisations for each nonlinear posterior sample:
    fullsamples, weights, y_preds = statmodel.get_weighted_samples(optresults['samples'], 2500)
    assert_allclose(weights, 1. / len(weights))
    assert fullsamples.shape == (2500 * len(optresults['samples']), 1)

    # verify that samples are normal distributed
    assert_allclose(fullsamples.mean(), y0[0], rtol=0.001)
    assert_allclose(fullsamples.std(), yerr0[0], rtol=0.001)

    samples, y_pred_samples = statmodel.resample(fullsamples, weights, y_preds)
    print(f'Obtained {len(samples)} equally weighted posterior samples')

    assert np.all(samples[:,0] == y_pred_samples[:,0])
    # verify that samples are normal distributed
    assert_allclose(samples.mean(), y0[0], rtol=0.001)
    assert_allclose(samples.std(), yerr0[0], rtol=0.001)

    # verify that samples are normal distributed
    assert_allclose(y_pred_samples.mean(), y0[0], rtol=0.001)
    assert_allclose(y_pred_samples.std(), yerr0[0], rtol=0.001)

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

    def linear_param_logprior(params):
        # 40 +- 2
        return -0.5 * ((params[:,0] - 40) / prior_sigma)**2
    
    np.random.seed(431)
    statmodel = OptNS(
        linear_param_names0, nonlinear_param_names0, compute_model_components0,
        nonlinear_param_transform0, linear_param_logprior,
        y, yerr**-2)

    # create a sampler from this
    optsampler = statmodel.ReactiveNestedSampler()
    optresults = optsampler.run(**ultranest_run_kwargs)
    fullsamples, weights, y_preds = statmodel.get_weighted_samples(optresults['samples'], 2500)
    assert not np.allclose(weights, 1. / len(weights)), 'expecting some reweighting!'
    assert fullsamples.shape == (2500 * len(optresults['samples']), 1)
    samples, y_pred_samples = statmodel.resample(fullsamples, weights, y_preds)
    assert np.all(samples[:,0] == y_pred_samples[:,0])

    # verify that samples are normal distributed
    assert_allclose(fullsamples.mean(), data_mean, rtol=0.001)
    assert_allclose(fullsamples.std(), measurement_sigma, rtol=0.001)

    assert_allclose(samples.mean(), expected_mean, rtol=0.001)
    assert_allclose(samples.std(), expected_std, rtol=0.001)

    assert_allclose(y_pred_samples.mean(), expected_mean, rtol=0.001)
    assert_allclose(y_pred_samples.std(), expected_std, rtol=0.001)


SNR = 1.0
Ndata = 10

# first we generate some mock data:

x = np.linspace(0, 10, Ndata)
A = 0 * x + 1
B = x
C = exp(-x / 5)
noise = 0.1 + 0 * x
model_linear = 1 * A + 1 * B
model_nonlinear = 1 * A + 1 * C

rng = np.random.RandomState(42)
data_linear = rng.normal(model_linear, noise)
data_nonlinear = rng.normal(model_nonlinear, noise)
data_nonlinear_poisson = rng.poisson(model_nonlinear)
X = np.transpose([A, B])

expected_res_OLS_mean = np.array([1.04791128, 0.99937897])
expected_res_OLS_cov = np.array([[ 3.45454545e-03, -4.90909091e-04], [-4.90909091e-04, 9.81818182e-05]])
# is less than 1 sigma off the true values
assert ((expected_res_OLS_mean - 1) / np.diag(expected_res_OLS_cov)**0.5 < 1.0).all()

def test_test_expectations():
    gm = GaussModel(data_linear, noise**-2, positive=False)
    gm.update_components(X)
    assert_allclose(expected_res_OLS_mean, LinearRegression(fit_intercept=False).fit(X, data_linear).coef_)
    assert_allclose(expected_res_OLS_mean, gm.norms())
    assert_allclose(expected_res_OLS_cov, np.linalg.inv(gm.XT_X))


# set up function which computes the various model components:
# the parameters are:
nonlinear_param_names = ['tau']
def compute_model_components_nonlinear(params):
    tau, = params
    return np.transpose([x * 0 + 1, np.exp(-x / tau)])

def compute_model_components_linear(params):
    return np.transpose([A, B])

# set up a prior transform for these nonlinear parameters
def nonlinear_param_transform(cube):
    return cube * 10

linear_param_names = ['A', 'B']

def linear_param_logprior_flat(params):
    assert params.shape[1] == len(linear_param_names)
    return np.where(params[:,0] < 10, 0, -1e300) + np.where(params[:,1] < 10, 0, -1e300)

def test_OLS():
    np.random.seed(431)
    statmodel = OptNS(
        ['A', 'B'], [], compute_model_components_linear,
        nonlinear_param_transform0, linear_param_logprior_flat,
        data_linear, noise**-2)
    optsampler = statmodel.ReactiveNestedSampler()
    optresults = optsampler.run(max_num_improvement_loops=0, frac_remain=0.5)

    fullsamples, weights, y_preds = statmodel.get_weighted_samples(optresults['samples'], 2500)
    assert np.allclose(weights, 1. / len(weights)), 'expecting no reweighting with flat prior'
    samples, y_pred_samples = statmodel.resample(fullsamples, weights, y_preds)

    # the result should agree with OLS
    print(samples.shape)
    mean = np.mean(samples, axis=0)
    print('mean:', mean)
    cov = np.cov(samples, rowvar=False)
    print('cov:', cov)
    assert_allclose(mean, expected_res_OLS_mean, rtol=0.001)
    assert_allclose(np.diag(cov)**0.5, np.diag(expected_res_OLS_cov)**0.5, rtol=0.001)
    assert_allclose(cov, expected_res_OLS_cov, atol=0.001)


def prior_transform_flat(cube):
    return cube * 10

def prior_transform_loguniform(cube):
    params = cube.copy()
    params[0] = 10**(cube[0] * 4 - 2)
    params[1] = 10**(cube[1] * 4 - 2)
    params[2] = 10 * cube[2]
    return params

def loglike(params):
    A, B, tau = params
    y_pred = A + B * np.exp(-x / tau)
    return -0.5 * np.sum(((y_pred - data_nonlinear) / noise)**2) + np.log(np.sqrt(2 * np.pi * noise**2)).sum()

def loglike_poisson(params):
    A, B, tau = params
    y_pred = A + B * np.exp(-x / tau)
    return scipy.stats.poisson(y_pred).logpmf(data_nonlinear_poisson).sum()

# prior log-probability density function for the linear parameters:
def linear_param_logprior_loguniform(params):
    logp = -np.log(params[:,0])
    logp += -np.log(params[:,1])
    logp += np.where(params[:,0] < 100, 0, -np.inf)
    logp += np.where(params[:,1] < 100, 0, -np.inf)
    logp += np.where(params[:,0] > 0.01, 0, -np.inf)
    logp += np.where(params[:,1] > 0.01, 0, -np.inf)
    return logp
    

def test_nonlinear_gauss_vs_full_nestedsampling():
    # run OptNS
    np.random.seed(123)
    statmodel = OptNS(
        ['A', 'B'], ['tau'], compute_model_components_nonlinear,
        nonlinear_param_transform, linear_param_logprior_flat,
        data_nonlinear, noise**-2)
    optsampler = statmodel.ReactiveNestedSampler()
    optresults = optsampler.run(**ultranest_run_kwargs)
    print(optresults['ncall'])
    fullsamples, weights, y_preds = statmodel.get_weighted_samples(optresults['samples'], 2500)
    samples, y_pred_samples = statmodel.resample(fullsamples, weights, y_preds)
    mean = np.mean(samples, axis=0)
    print('mean:', mean)
    std = np.std(samples, axis=0)
    print('std:', std)
    cov = np.cov(samples, rowvar=False)
    print('cov:', cov)

    # run full nested sampling
    np.random.seed(234)
    refrun_sampler = ReactiveNestedSampler(['A', 'B', 'tau'], loglike, transform=prior_transform_flat)
    refrun_result = refrun_sampler.run(**ultranest_run_kwargs)
    ref_mean = np.mean(refrun_result['samples'], axis=0)
    print('ref_mean:', ref_mean)
    ref_std = np.std(refrun_result['samples'], axis=0)
    print('ref_std:', ref_std)
    ref_cov = np.cov(refrun_result['samples'], rowvar=False)
    print('ref_cov:', ref_cov)

    ax = plt.figure(figsize=(15, 6)).gca()
    statmodel.posterior_predictive_check_plot(ax, samples[:100])
    plt.legend()
    plt.savefig('test_gauss_ppc.pdf')
    plt.close()

    fig = corner.corner(samples, titles=['A', 'B', 'tau'], labels=['A', 'B', 'tau'], show_titles=True, plot_datapoints=False, plot_density=False)
    corner.corner(refrun_result['samples'], fig=fig, color='red', truths=[1, 1, 5])
    plt.savefig('test_gauss_corner.pdf')
    plt.close()

    # check agreement
    assert_allclose(std, ref_std, rtol=0.05)
    assert_allclose(mean[0], ref_mean[0], atol=std[0] * 0.06)
    assert_allclose(mean[1], ref_mean[1], atol=std[1] * 0.06)
    assert_allclose(mean[2], ref_mean[2], atol=std[2] * 0.15)
    assert_allclose(cov, ref_cov, atol=0.01, rtol=0.1)
    """
    mean: [0.81558542 1.12452735 5.90837637]
    ref_mean: [0.82504667 1.11770239 5.78328028]
    std: [0.18848388 0.18708056 1.97166544]
    ref_std: [0.18139405 0.17909643 1.9150434 ]
    cov: [[ 0.03552619 -0.03201207 -0.34346579]
     [-0.03201207  0.03499915  0.2715496 ]
     [-0.34346579  0.2715496   3.88746632]]
    ref_cov: [[ 0.03291193 -0.02936161 -0.3219523 ]
     [-0.02936161  0.03208346  0.25032964]
     [-0.3219523   0.25032964  3.66829788]]
    """

    # check that OptNS has fewer evaluations
    print(optresults['ncall'], refrun_result['ncall'])
    assert optresults['ncall'] < refrun_result['ncall'] // 4 - 100  # four times faster

mem = joblib.Memory('.')

@mem.cache
def get_full_nested_sampling_run(seed):
    np.random.seed(seed)
    refrun_sampler = ReactiveNestedSampler(['A', 'B', 'tau'], loglike_poisson, transform=prior_transform_loguniform)
    return refrun_sampler.run(**ultranest_run_kwargs, min_num_live_points=2000)

def test_nonlinear_poisson_vs_full_nestedsampling():
    # run OptNS
    np.random.seed(123)
    statmodel = OptNS(
        ['A', 'B'], ['tau'], compute_model_components_nonlinear,
        nonlinear_param_transform, linear_param_logprior_loguniform,
        data_nonlinear_poisson)
    optsampler = statmodel.ReactiveNestedSampler()
    optresults = optsampler.run(**ultranest_run_kwargs)
    print(optresults['ncall'])
    #i = np.argsort(optresults['samples'][:40, 0])
    fullsamples, weights, y_preds = statmodel.get_weighted_samples(optresults['samples'], 1000)
    print('fullsamples', fullsamples.shape)
    samples, y_pred_samples = statmodel.resample(fullsamples, weights, y_preds)
    print('samples', samples.shape)
    mean = np.mean(samples, axis=0)
    print('mean:', mean)
    std = np.std(samples, axis=0)
    print('std:', std)
    cov = np.cov(samples, rowvar=False)
    print('cov:', cov)

    # run full nested sampling
    refrun_result = get_full_nested_sampling_run(12345)
    ref_mean = np.mean(refrun_result['samples'], axis=0)
    print('ref_mean:', ref_mean)
    ref_std = np.std(refrun_result['samples'], axis=0)
    print('ref_std:', ref_std)
    ref_cov = np.cov(refrun_result['samples'], rowvar=False)
    print('ref_cov:', ref_cov)

    ax = plt.figure(figsize=(15, 6)).gca()
    statmodel.posterior_predictive_check_plot(ax, samples[:100])
    plt.legend()
    plt.savefig('test_poisson_ppc.pdf')
    plt.close()

    refrun_samples = refrun_result['samples'].copy()
    samples[:,0:2] = np.log10(samples[:,0:2] + 0.001)
    fullsamples[:,:2] = np.log10(fullsamples[:,0:2] + 0.001)
    refrun_samples[:,0:2] = np.log10(refrun_samples[:,0:2] + 0.001)
    fig = corner.corner(samples, titles=['logA', 'logB', 'tau'], labels=['logA', 'logB', 'tau'], show_titles=True, plot_datapoints=False, plot_density=False)
    #fig = corner.corner(fullsamples, color='navy', titles=['logA', 'logB', 'tau'], labels=['logA', 'logB', 'tau'], show_titles=True, plot_datapoints=False, plot_density=False, weights=weights)
    corner.corner(refrun_samples, fig=fig, color='red', truths=[0, 0, 5], plot_datapoints=False, plot_density=False, )
    plt.savefig('test_poisson_corner.pdf')
    plt.close()
    
    

    #fig = corner.corner(samples, titles=['A', 'B', 'tau'], labels=['A', 'B', 'tau'], show_titles=True, plot_datapoints=False, plot_density=False)
    #corner.corner(fullsamples, fig=fig, color='navy', plot_datapoints=False, plot_density=False, weights=weights)
    #corner.corner(refrun_result['samples'], fig=fig, color='red', truths=[1, 1, 5], plot_datapoints=False, plot_density=False)
    """axes = np.array(fig.axes).reshape((3, 3))
    plt.sca(axes[2,2])
    axes[2,2].clear()
    kwargs = dict(bins=np.arange(0, 10, 1.0), density=True, histtype='step')
    axes[2,2].hist(samples[:,2], **kwargs, color='k', label='resampled')
    axes[2,2].hist(fullsamples[:,2], **kwargs, color='green', label='proposals')
    axes[2,2].hist(fullsamples[:,2], **kwargs, color='navy', label='weighted samples', weights=weights)
    axes[2,2].hist(optresults['samples'][:,-1], **kwargs, color='purple', label='profile samples')
    axes[2,2].hist(refrun_result['samples'][:,2], **kwargs, color='red', label='UltraNest')
    axes[2,2].set_yticks([])
    axes[2,2].set_xlim(0, 10)
    axes[2,2].legend()"""
    #plt.savefig('test_poisson_corner.pdf')

    # check agreement
    assert_allclose(std, ref_std, rtol=0.2)
    assert_allclose(mean[0], ref_mean[0], atol=std[0] * 0.2)
    assert_allclose(mean[1], ref_mean[1], atol=std[1] * 0.2)
    assert_allclose(mean[2], ref_mean[2], atol=std[2] * 0.2)
    assert_allclose(cov, ref_cov, atol=0.01, rtol=0.1)

    """
    std: [0.54924376 0.9545564  3.00385723]
    ref_std: [0.48116927 1.18044312 3.01782731]

    mean: [0.93985552 0.62048913 5.99273564]
    ref_mean: [0.95835405 1.58101927 4.60394989]

    cov: [[ 0.3040629  -0.41073453 -0.3591285 ]
     [-0.41073453  0.91840948  0.41455408]
     [-0.3591285   0.41455408  9.09477061]]
    ref_cov: [[ 0.23163266 -0.16238455 -0.49514391]
     [-0.16238455  1.39410078 -0.98509254]
     [-0.49514391 -0.98509254  9.11156139]]
    """

    # check that OptNS has fewer evaluations
    print(optresults['ncall'], refrun_result['ncall'])
    assert optresults['ncall'] < refrun_result['ncall'] // 5 - 100  # five times faster

