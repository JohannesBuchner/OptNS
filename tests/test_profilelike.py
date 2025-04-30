import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.linear_model import LinearRegression
from optns.profilelike import poisson_negloglike, poisson_negloglike_grad, poisson_negloglike_hessian, ComponentModel, GaussianPrior, gauss_importance_sample_stable
import scipy.stats
from scipy.special import factorial
from numpy.testing import assert_allclose

def test_gauss_importance_sample_stable_fallback():
    rng = np.random.RandomState(seed=42)
    mean = np.array([0.0, 0.0])
    # Singular covariance matrix (not full rank)
    covariance = np.array([[1.0, 1.0],
                           [1.0, 1.0]])
    samples, logpdf = gauss_importance_sample_stable(
        mean, covariance, size=5, rng=rng)
    assert samples.shape == (5, 2)
    # Check that logpdf runs and returns array of correct length
    test_input = np.array([[0.0, 0.0], [1.0, 1.0]])
    log_probs = logpdf(test_input)
    assert log_probs.shape == (2,)
    assert np.all(np.isfinite(log_probs))

def test_poisson_negloglike_lowcount():
    counts = np.array([0, 1, 2, 3])
    X = np.ones((4, 3))
    lognorms = np.array([-1e100, 10, 0.1])
    logl = -poisson_negloglike(lognorms, X, counts)
    assert np.isfinite(logl), logl
    assert_allclose(
        logl - np.log(factorial(counts)).sum(),
        poisson.logpmf(counts, exp(lognorms) @ X.T).sum()
    )
    logl2 = -poisson_negloglike(np.zeros(3), X, counts)
    assert np.isfinite(logl2), logl2
    # should be near 1
    assert np.abs(logl2) < 10
    np.testing.assert_allclose(
        logl2 - np.log(factorial(counts)).sum(),
        poisson.logpmf(counts, np.ones(3) @ X.T).sum()
    )

def test_poisson_negloglike_highcount():
    counts = np.array([10000, 10000])
    X = np.ones((2, 1))
    logl2 = -poisson_negloglike(np.array([-10]), X, counts)
    assert np.isfinite(logl2), logl2
    logl3 = -poisson_negloglike(np.log([10000]), X, counts)
    assert np.isfinite(logl3), logl3
    
    assert logl3 > logl2

def test_poisson_negloglike_grad_highcount():
    counts = np.array([10000])
    X = np.ones((1, 1))
    logl = -poisson_negloglike(np.log([10000]), X, counts)
    logl1 = -poisson_negloglike(np.log([11000]), X, counts)
    logl2 = -poisson_negloglike(np.log([9000]), X, counts)
    assert logl > logl1
    assert logl > logl2
    grad1 = -poisson_negloglike_grad(np.log([11000]), X, counts)
    print(grad1)
    assert grad1 < 0
    grad2 = -poisson_negloglike_grad(np.log([9000]), X, counts)
    print(grad2)
    assert grad2 > 0
    hess = poisson_negloglike_hessian(np.log([10000]), X, counts)
    print(hess)
    hess1 = poisson_negloglike_hessian(np.log([11000]), X, counts)
    print(hess1)
    hess2 = poisson_negloglike_hessian(np.log([9000]), X, counts)
    print(hess2)

def test_gauss():
    x = np.linspace(0, 10, 400)
    A = 0 * x + 1
    B = x
    C = np.sin(x)**2
    model = 3 * A + 0.5 * B + 5 * C
    noise = 0.5 + 0.1 * x

    rng = np.random.RandomState(42)
    data = rng.normal(model, noise)

    X = np.transpose([A, B, C])
    y = data
    sample_weight = noise**-2
    statmodel = ComponentModel(3, data, flat_invvar=sample_weight)
    statmodel.update_components(X)
    logl = statmodel.loglike()
    norms_inferred = statmodel.norms()
    np.testing.assert_allclose(norms_inferred, [2.87632905, 0.52499782, 5.08684032])
    reg = LinearRegression(positive=True, fit_intercept=False)
    reg.fit(X, y, sample_weight)
    y_model = X @ reg.coef_
    loglike_manual_prefactor = np.sum(np.log(1. / np.sqrt(2 * np.pi * noise**2)))
    np.testing.assert_allclose(loglike_manual_prefactor, statmodel.loglike_prefactor)
    loglike_manual = -0.5 * np.sum((y - y_model)**2 * sample_weight) + loglike_manual_prefactor
    np.testing.assert_allclose(norms_inferred, reg.coef_)
    np.testing.assert_allclose(logl, loglike_manual)
    samples, _, logl_samples = statmodel.sample(100000, rng)
    # samples should be centered at reg.coef
    print(X.shape, samples.shape, logl_samples.shape)
    print(samples.mean(axis=0), reg.coef_)
    np.testing.assert_allclose(samples.mean(axis=0), reg.coef_, atol=0.0003)
    #y_model_samples = np.einsum('ji,ki->kj', X, samples)
    y_model_samples = samples @ X.T
    loglike_manual_samples = -0.5 * np.sum((y - y_model_samples)**2 * sample_weight, axis=1) + loglike_manual_prefactor
    print(loglike_manual_samples.shape, logl_samples.shape, logl_samples[0], loglike_manual_samples[0])
    np.testing.assert_allclose(logl_samples, loglike_manual_samples)
    assert np.all(samples > 0)
    # plot 
    plt.figure(figsize=(15, 6))
    plt.plot(x, model)
    plt.errorbar(x, data, noise, capsize=2, elinewidth=0.5, linestyle=' ')
    for sample in samples[::400]:
        plt.plot(x, X @ sample, ls='-', lw=1, alpha=0.5)
        np.testing.assert_allclose(sample, reg.coef_, atol=0.1, rtol=0.2)
    plt.savefig('testgausssampling.pdf')
    plt.close()


def test_trivial_OLS():
    y = np.array([42.0])
    yerr = np.array([1.0])
    X = np.transpose([[1.]])
    statmodel = ComponentModel(1, y, flat_invvar=yerr**-2)
    statmodel.update_components(X)
    chi2 = statmodel.chi2()
    assert chi2 == 0
    norms_inferred = statmodel.norms()
    assert norms_inferred == 42.0


def test_poisson_verylowcount():
    x = np.ones(1)
    A = 0 * x + 1
    rng = np.random.RandomState(42)
    X = np.transpose([A])
    for ncounts in 0, 1, 2, 3, 4, 5, 10, 20, 40, 100:
        data = np.array([ncounts])
        statmodel = ComponentModel(1, data)
        statmodel.update_components(X)
        samples, loglike_proposal, loglike_target = statmodel.sample(1000000, rng)
        assert np.all(samples > 0)
        Nsamples = len(samples)
        assert samples.shape == (Nsamples, 1), samples.shape
        assert loglike_proposal.shape == (Nsamples,)
        assert loglike_target.shape == (Nsamples,)
        # plot 
        bins = np.linspace(0, samples.max(), 200)
        plt.figure()
        plt.hist(samples[:,0], density=True, histtype='step', bins=bins, color='grey', ls='--')
        weight = exp(loglike_target - loglike_proposal - np.max(loglike_target - loglike_proposal))
        weight /= weight.sum()
        N, _, _ = plt.hist(samples[:,0], density=True, weights=weight, histtype='step', bins=bins, color='k')
        logl = poisson.logpmf(ncounts, bins)
        plt.plot(bins, np.exp(logl - logl.max()) * N.max(), drawstyle='steps-mid')
        plt.savefig(f'testpoissonprofilelike{ncounts}.pdf')
        plt.close()


def test_poisson_lowcount():
    x = np.linspace(0, 10, 400)
    A = 0 * x + 1
    B = x
    C = np.sin(x)**2
    model = 3 * A + 0.5 * B + 5 * C

    rng = np.random.RandomState(42)
    data = rng.poisson(model)

    X = np.transpose([A, B, C])
    def minfunc(lognorms):
        lam = np.exp(lognorms) @ X.T
        loglike = data * np.log(lam) - lam
        # print('  ', lognorms, loglike.sum())
        return -loglike.sum()

    x0 = np.log(np.median((data.reshape((-1, 1)) + 0.1) / (X + 0.1), axis=0))
    res = minimize(minfunc, x0, method='Nelder-Mead', options=dict(fatol=1e-10, maxfev=10000))
    norms_expected = np.exp(res.x)

    statmodel = ComponentModel(3, data)
    statmodel.update_components(X)
    logl = statmodel.loglike()
    logl_expected = -poisson_negloglike(res.x, X, data)
    assert np.isclose(logl, logl_expected), (logl, logl_expected)
    norms_inferred = statmodel.norms()
    np.testing.assert_allclose(norms_inferred, [2.71413583, 0.46963565, 5.45321002], atol=1e-3)
    np.testing.assert_allclose(norms_inferred, norms_expected, atol=0.001)
    samples, loglike_proposal, loglike_target = statmodel.sample(100000, rng)
    assert np.all(samples > 0)
    Nsamples = len(samples)
    assert samples.shape == (Nsamples, 3), samples.shape
    assert loglike_proposal.shape == (Nsamples,)
    assert loglike_target.shape == (Nsamples,)
    # plot 
    plt.figure(figsize=(15, 6))
    plt.plot(x, model)
    plt.scatter(x, data)
    for sample in samples[::4000]:
        plt.plot(x, X @ sample, ls='-', lw=1, alpha=0.5, color='k')
        #np.testing.assert_allclose(sample, reg.coef_, atol=0.1, rtol=0.2)
    
    weight = exp(loglike_target - loglike_proposal - np.max(loglike_target - loglike_proposal))
    print(weight, weight.min(), weight.max(), weight.mean())
    weight /= weight.sum()
    rejection_sampled_indices = rng.choice(len(samples), p=weight, size=40)
    for sample in samples[rejection_sampled_indices,:]:
        plt.plot(x, X @ sample, ls='-', lw=1, alpha=0.5, color='r')
    
    plt.savefig('testpoissonsampling.pdf')
    plt.close()



def test_poisson_highcount():
    x = np.linspace(0, 10, 400)
    A = 0 * x + 1
    B = x
    C = np.sin(x)**2
    model = 30 * A + 5 * B + 50 * C

    rng = np.random.RandomState(42)
    data = rng.poisson(model)

    X = np.transpose([A, B, C])
    def minfunc(lognorms):
        lam = np.exp(lognorms) @ X.T
        loglike = data * np.log(lam) - lam
        # print('  ', lognorms, loglike.sum())
        return -loglike.sum()

    x0 = np.log(np.median((data.reshape((-1, 1)) + 0.1) / (X + 0.1), axis=0))
    #res = minimize(minfunc, x0, method='L-BFGS-B', options=dict(ftol=1e-10, maxfun=10000))
    res = minimize(minfunc, x0, method='Nelder-Mead', options=dict(fatol=1e-10, maxfev=10000))
    norms_expected = np.exp(res.x)

    statmodel = ComponentModel(3, data)
    statmodel.update_components(X)
    logl = statmodel.loglike()
    logl_expected = -poisson_negloglike(res.x, X, data)
    assert np.isclose(logl, logl_expected), (logl, logl_expected)
    norms_inferred = statmodel.norms()
    np.testing.assert_allclose(norms_inferred, norms_expected, atol=0.001)
    np.testing.assert_allclose(norms_inferred, [29.940845,  4.736076, 51.151115], atol=0.001)
    samples, loglike_proposal, loglike_target = statmodel.sample(100000, rng)
    assert np.all(samples > 0)
    Nsamples = len(samples)
    assert samples.shape == (Nsamples, 3), samples.shape
    assert loglike_proposal.shape == (Nsamples,)
    assert loglike_target.shape == (Nsamples,)
    # plot 
    plt.figure(figsize=(15, 6))
    plt.plot(x, model)
    plt.scatter(x, data)
    for sample in samples[::4000]:
        plt.plot(x, X @ sample, ls='-', lw=1, alpha=0.5, color='k')
        #np.testing.assert_allclose(sample, reg.coef_, atol=0.1, rtol=0.2)
    
    weight = exp(loglike_target - loglike_proposal - np.max(loglike_target - loglike_proposal))
    print(weight, weight.min(), weight.max(), weight.mean())
    weight /= weight.sum()
    rejection_sampled_indices = rng.choice(len(samples), p=weight, size=40)
    for sample in samples[rejection_sampled_indices,:]:
        plt.plot(x, X @ sample, ls='-', lw=1, alpha=0.5, color='r')
    
    plt.savefig('testpoissonsampling2.pdf')
    plt.close()



def test_poisson_loglike():
    Ns = [10, 40, 100, 400, 1000, 4000, 10000]
    SNRs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    for Ndata in Ns:
        x = np.linspace(0, 10, Ndata)
        A = 0 * x + 1
        B = x
        C = np.sin(x + 2)**2
        for SNR in SNRs:
            model = (3 * A + 0.5 * B + 5 * C) * SNR
            X = np.transpose([A, B, C])
            rng = np.random.RandomState(Ndata)
            data = rng.poisson(model)
            if data.sum() == 0: continue
            statmodel = ComponentModel(3, data)
            statmodel.update_components(X)
            norms_inferred = statmodel.norms()
            assert np.isfinite(norms_inferred).all(), norms_inferred
            logl = statmodel.loglike()
            assert np.isfinite(logl), logl

def test_poisson_component_somezero():
    Ndata = 40
    x = np.linspace(0, 10, Ndata)
    A = 0 * x + 1
    B = np.where(x > 8, x, 0)
    C = np.sin(x + 2)**2
    model = 3 * A + 0.5 * B + 5 * C
    X = np.transpose([A, B, C])
    rng = np.random.RandomState(Ndata)
    data = rng.poisson(model)
    assert data.sum() > 0
    statmodel = ComponentModel(3, data)
    statmodel.update_components(X)
    norms_inferred = statmodel.norms()
    assert np.isfinite(norms_inferred).all(), norms_inferred
    logl = statmodel.loglike()
    assert np.isfinite(logl), logl

def test_poisson_component_zero():
    Ndata = 40
    x = np.linspace(0, 10, Ndata)
    A = 0 * x + 1
    B = x * 0
    C = np.sin(x + 2)**2
    model = 3 * A + 0.5 * B + 5 * C
    X = np.transpose([A, B, C])
    rng = np.random.RandomState(Ndata)
    data = rng.poisson(model)
    assert data.sum() > 0
    statmodel = ComponentModel(3, data)
    try:
        statmodel.update_components(X)
        raise Exception()
    except AssertionError:
        pass


def test_poisson_components_identical():
    Ndata = 40
    x = np.linspace(0, 10, Ndata)
    A = 0 * x + 1
    B = x
    C = B
    model = 3 * A + 0.5 * B + 5 * C
    X = np.transpose([A, B, C])
    rng = np.random.RandomState(Ndata)
    data = rng.poisson(model)
    assert data.sum() > 0
    statmodel = ComponentModel(3, data)
    statmodel.update_components(X)
    assert statmodel.norms()[2] == 0
    logl = statmodel.loglike()
    assert np.isfinite(logl), logl


def test_gauss_components_identical():
    x = np.linspace(0, 10, 400)
    A = 0 * x + 1
    B = x
    C = B
    model = 3 * A + 0.5 * B + 5 * C
    noise = 0.5 + 0.1 * x

    rng = np.random.RandomState(42)
    data = rng.normal(model, noise)

    X = np.transpose([A, B, C])
    sample_weight = noise**-2
    statmodel = ComponentModel(3, data, flat_invvar=sample_weight)
    statmodel.update_components(X)
    assert statmodel.cond > 1e6
    assert statmodel.norms()[2] == 0


def test_gaussian_prior():
    means = np.array([1.23, 4.56])
    covs = np.array([[1, 0], [0, 0.01]])
    gauss = GaussianPrior(means, covs)
    rva = scipy.stats.norm(1.23, 1)
    rvb = scipy.stats.norm(4.56, 0.1)
    logpdfa = rva.logpdf(2.0)
    logpdfb = rvb.logpdf(4.0)
    logpdf = -gauss.neglogprob([2.0, 4.0])
    print(logpdfa, logpdfb, logpdf)
    assert_allclose(logpdf, logpdfa + logpdfb)
    assert_allclose(logpdf, gauss.logprob_many(np.array([[2.0, 4.0]]))[0])

def test_gaussian_prior2():
    means = np.array([1.23, 4.56])
    covs = np.array([[1, 0.02], [0.02, 0.1]])
    #covs = np.array([[1, 0.0], [0.0, 0.1]])
    gauss = GaussianPrior(means, covs)
    rv = scipy.stats.multivariate_normal(means, covs)
    samples = np.random.multivariate_normal(means, covs, size=100)
    for sample in samples:
        print('sample:', sample)
        assert_allclose(rv.logpdf(sample), -gauss.neglogprob(sample))
    logpdfs = rv.logpdf(samples)
    assert_allclose(logpdfs, gauss.logprob_many(samples))
    
    assert_allclose([0, 0], gauss.grad(means))
    invcov = np.linalg.inv(covs)
    assert_allclose(invcov, gauss.hessian(means))
    for sample in samples[:3]:
        assert_allclose(gauss.grad(sample), invcov @ (sample - means))
        assert_allclose(gauss.hessian(sample), invcov)
