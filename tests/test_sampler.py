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

    # check agreement
    assert_allclose(std, ref_std, rtol=0.05)
    assert_allclose(mean[0], ref_mean[0], atol=std[0] * 0.1)
    assert_allclose(mean[1], ref_mean[1], atol=std[1] * 0.1)
    assert_allclose(mean[2], ref_mean[2], atol=std[2] * 0.12)
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

def additive_invvar_function(params):
    return 1. / (noise**2 + params[-1]**2)

def fractional_invvar_function(params):
    return 1. / (noise**2 + (data_linear * (1. + params[-1]))**2)

def test_nonlinear_gauss_variable_error():
    np.random.seed(123)
    statmodel = OptNS(
        ['A', 'B'], ['tau'], compute_model_components_nonlinear,
        nonlinear_param_transform, linear_param_logprior_flat,
        data_linear, noise**-2, positive=False)
    optsampler = statmodel.ReactiveNestedSampler()
    optresults = optsampler.run(**ultranest_run_kwargs)
    optsampler.print_results()
    fullsamples, weights, y_preds = statmodel.get_weighted_samples(optresults['samples'], 100)
    samples, y_pred_samples = statmodel.resample(fullsamples, weights, y_preds)
    mean = np.mean(samples, axis=0)
    print('mean:', mean)
    std = np.std(samples, axis=0)
    print('std:', std)
    print('logz:', optresults['logz'])

    np.random.seed(123)
    statmodel2 = OptNS(
        ['A', 'B'], ['tau', 'fracsys'], lambda params: compute_model_components_nonlinear(params[:1]),
        nonlinear_param_transform, linear_param_logprior_flat,
        data_linear, noise**-2, compute_invvar=fractional_invvar_function, positive=False)
    optsampler2 = statmodel2.ReactiveNestedSampler()
    optresults2 = optsampler2.run(**ultranest_run_kwargs)
    optsampler2.print_results()
    fullsamples2, weights2, y_preds2 = statmodel2.get_weighted_samples(optresults2['samples'], 100)
    samples2, y_pred_samples2 = statmodel2.resample(fullsamples2, weights2, y_preds2)
    mean2 = np.mean(samples2, axis=0)
    print('mean2:', mean2)
    std2 = np.std(samples2, axis=0)
    print('std2:', std2)
    print('logz2:', optresults2['logz'])
    assert (samples2[:,-1] > 0.05).mean() > 0.5, 'needed extra noise'
    assert (std2[:-1] > std).all()

    np.random.seed(123)
    statmodel3 = OptNS(
        ['A', 'B'], ['tau', 'addnoise'], lambda params: compute_model_components_nonlinear(params[:1]),
        nonlinear_param_transform, linear_param_logprior_flat,
        data_linear, noise**-2, compute_invvar=additive_invvar_function, positive=False)
    optsampler3 = statmodel3.ReactiveNestedSampler()
    optresults3 = optsampler3.run(**ultranest_run_kwargs)
    optsampler3.print_results()
    fullsamples3, weights3, y_preds3 = statmodel3.get_weighted_samples(optresults3['samples'], 100)
    samples3, y_pred_samples3 = statmodel3.resample(fullsamples3, weights3, y_preds3)
    mean3 = np.mean(samples3, axis=0)
    print('mean3:', mean3)
    std3 = np.std(samples3, axis=0)
    print('std3:', std3)
    print('logz3:', optresults3['logz'])
    assert (samples3[:,-1] > 0.1).mean() > 0.95, 'needed extra noise'
    assert (std3[:-1] > std).all()
    assert optresults['logz'] < optresults2['logz'] < optresults3['logz']



def test_GP():
    import george

    def true_model(t, amp, location, log_sigma2):
        return amp * np.exp(-0.5 * (t - location)**2 * np.exp(-log_sigma2))

    np.random.seed(1234)

    def generate_data(params, N, rng=(-5, 5)):
        gp = george.GP(0.1 * george.kernels.ExpSquaredKernel(3.3))
        t = rng[0] + np.diff(rng) * np.sort(np.random.rand(N))
        y = gp.sample(t) + true_model(t, **params)
        yerr = 0.05 + 0.05 * np.random.rand(N)
        y += yerr * np.random.randn(N)
        return t, y, yerr

    truth = dict(amp=-1.0, location=0.1, log_sigma2=np.log(0.4))
    t, y, yerr = generate_data(truth, 50)
    tdata = t

    plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
    plt.ylabel(r"$y$")
    plt.xlabel(r"$t$")
    plt.xlim(-5, 5)
    plt.savefig('test_GP_data.pdf')
    plt.close()


    gp = george.GP(np.var(y) * george.kernels.Matern32Kernel(10.0))
    gp.compute(t, yerr)
    linear_param_names = ['C', 'amp']
    nonlinear_param_names = ['location', 'lnvar'] + list(gp.get_parameter_names())
    def compute_model_components(params):
        # handle deterministic parameters
        l, lnvar = params[:2]
        X = np.transpose([1.0 + 0 * t, true_model(t, 1.0, l, lnvar)])
        # the rest are GP parameters, which we update:
        gp.set_parameter_vector(params[2:])
        gp.compute(tdata, yerr)
        return X
    def nonlinear_param_transform(cube):
        params = cube.copy()
        params[0] = 20 * cube[0] - 10  # C
        params[1] = 6 * cube[1] - 3    # amp
        # GP parameter priors:
        params[2:] = cube[2:] * 20 - 10
        return params
    def linear_param_logprior(params):
        return 0

    np.random.seed(123)
    statmodel = OptNS(
        linear_param_names, nonlinear_param_names, compute_model_components,
        nonlinear_param_transform, linear_param_logprior,
        y, gp=gp, positive=False)
    optsampler = statmodel.ReactiveNestedSampler(log_dir='test_GP-run', resume='overwrite')
    optresults = optsampler.run(**ultranest_run_kwargs)
    optsampler.print_results()
    optsampler.plot()
    fullsamples, weights, y_preds = statmodel.get_weighted_samples(optresults['samples'], 100)
    samples, y_pred_samples = statmodel.resample(fullsamples, weights, y_preds)

    corner.corner(
        samples,
        titles=linear_param_names + nonlinear_param_names,
        labels=linear_param_names + nonlinear_param_names,
        show_titles=True, plot_datapoints=False, plot_density=False,
        truths=[0, -1, 0.1, np.log(0.4), np.log(0.1), np.log(3.3)])
    plt.savefig('test_GP_corner.pdf')
    plt.close()

    ax = plt.figure(figsize=(15, 6)).gca()
    plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
    plt.ylabel(r"$y$")
    plt.xlabel(r"$t$")
    plt.xlim(-5, 5)
    t = np.linspace(-5, 5, 500)
    colors = ['orange', 'pink']
    for i, (sample, y_pred_sample) in enumerate(zip(samples[:40], y_pred_samples[:40])):
        norms = sample[:len(statmodel.linear_param_names)]
        nonlinear_params = sample[len(statmodel.linear_param_names):]
        X = statmodel.compute_model_components(nonlinear_params)
        for j, norm in enumerate(norms):
            if i == 0:
                l, = ax.plot(t, norm * X[:,j], alpha=0.2, lw=0.5, label=statmodel.linear_param_names[j])
                colors.append(l.get_color())
            else:
                l, = ax.plot(t, norm * X[:,j], alpha=0.2, lw=0.5, color=colors[j])
        y_pred = norms @ X.T
        plt.plot(t, y_pred + gp.sample_conditional(y - y_pred_sample, t), color="#4682b4", alpha=0.3, label='total' if i == 0 else None)
    plt.legend()
    plt.savefig('test_GP_ppc.pdf')
    plt.close()


