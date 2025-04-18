import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.linear_model import LinearRegression
from optns.profilelike import poisson_negloglike, ComponentModel
from scipy.special import factorial
from numpy.testing import assert_allclose


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
    logl = statmodel.loglike_gauss(X)
    norms_inferred = statmodel.norms_gauss(X)
    np.testing.assert_allclose(norms_inferred, [2.87632905, 0.52499782, 5.08684032])
    reg = LinearRegression(positive=True, fit_intercept=False)
    reg.fit(X, y, sample_weight)
    y_model = X @ reg.coef_
    loglike_manual = -0.5 * np.sum((y - y_model)**2 * sample_weight)
    np.testing.assert_allclose(norms_inferred, reg.coef_)
    np.testing.assert_allclose(logl, loglike_manual)
    samples, _, _ = statmodel.sample_gauss(X, 10000, rng)
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


def test_poisson_verylowcount():
    x = np.ones(1)
    A = 0 * x + 1
    rng = np.random.RandomState(42)
    X = np.transpose([A])
    for ncounts in 0, 1, 2, 3, 4, 5, 10, 20, 40, 100:
        data = np.array([ncounts])
        statmodel = ComponentModel(1, data)
        samples, loglike_proposal, loglike_target = statmodel.sample_poisson(X, 1000000, rng)
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
    logl = statmodel.loglike_poisson(X)
    logl_expected = -poisson_negloglike(res.x, X, data)
    assert np.isclose(logl, logl_expected), (logl, logl_expected)
    norms_inferred = statmodel.norms_poisson(X)
    np.testing.assert_allclose(norms_inferred, [2.71413583, 0.46963565, 5.45321002], atol=1e-3)
    np.testing.assert_allclose(norms_inferred, norms_expected, atol=0.001)
    samples, loglike_proposal, loglike_target = statmodel.sample_poisson(X, 100000, rng)
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
    logl = statmodel.loglike_poisson(X)
    logl_expected = -poisson_negloglike(res.x, X, data)
    assert np.isclose(logl, logl_expected), (logl, logl_expected)
    norms_inferred = statmodel.norms_poisson(X)
    np.testing.assert_allclose(norms_inferred, norms_expected, atol=0.001)
    np.testing.assert_allclose(norms_inferred, [29.940845,  4.736076, 51.151115], atol=0.001)
    samples, loglike_proposal, loglike_target = statmodel.sample_poisson(X, 100000, rng)
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
            norms_inferred = statmodel.norms_poisson(X)
            assert np.isfinite(norms_inferred).all(), norms_inferred
            logl = statmodel.loglike_poisson(X)
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
    norms_inferred = statmodel.norms_poisson(X)
    assert np.isfinite(norms_inferred).all(), norms_inferred
    logl = statmodel.loglike_poisson(X)
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
        statmodel.norms_poisson(X)
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
    assert statmodel.norms_poisson(X)[2] == 0
    logl = statmodel.loglike_poisson(X)
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
    y = data
    sample_weight = noise**-2
    statmodel = ComponentModel(3, data, flat_invvar=sample_weight)
    assert statmodel.norms_gauss(X)[2] == 0

