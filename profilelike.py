import numpy as np
from numpy import exp, log
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from sklearn.linear_model import LinearRegression


def poisson_negloglike(lognorms, X, counts):
    """Compute negative log-likelihood of a Poisson distribution.

    Parameters
    ----------
    lognorms: array
        logarithm of normalisations
    X: array
        transposed list of the model component vectors.
    counts: array
        non-negative integers giving the observed counts.

    Returns
    -------
    negloglike: float
        negative log-likelihood, neglecting the `1/fac(counts)` constant.
    """
    lam = exp(lognorms) @ X.T
    loglike = counts * log(lam) - lam
    return -loglike.sum()


def test_poisson_negloglike_lowcount():
    counts = np.array([0, 1, 2, 3])
    X = np.ones((4, 3))
    lognorms = np.array([-1e100, 10, 0.1])
    logl = -poisson_negloglike(lognorms, X, counts)
    assert np.isfinite(logl), logl
    from scipy.special import factorial
    from scipy.stats import poisson
    np.testing.assert_allclose(
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


class ComponentModel:
    """Generalized Additive Model.

    Defines likelihoods for observed data,
    given arbitrary components which are
    linearly added with non-negative normalisations.
    """

    def __init__(self, Ncomponents, flat_data, flat_invvar=None):
        """Initialise.

        Parameters
        ----------
        Ncomponents: int
            number of model components
        flat_data: array
            array of observed data. For the Poisson likelihood functions,
            must be non-negative integers.
        flat_invvar: None|array
            For the Poisson likelihood functions, None.
            For the Gaussian likelihood function, the inverse variance,
            `1 / (standard_deviation)^2`, where standard_deviation
            are the measurement uncertainties.
        """
        (self.Ndata,) = flat_data.shape
        self.flat_data = flat_data
        if flat_invvar is not None:
            self.flat_invvar = flat_invvar
            self.invvar_matrix = np.diag(self.flat_invvar)
        self.Ncomponents = Ncomponents
        self.poisson_guess_data_offset = 0.1
        self.poisson_guess_model_offset = 0.1
        self.minimize_kwargs = dict(method="L-BFGS-B")
        self.cond_threshold = 1e6
        self.gauss_reg = LinearRegression(positive=True, fit_intercept=False)

    def loglike_poisson_optimize(self, component_shapes):
        """Optimize the normalisations assuming a Poisson Additive Model.

        Parameters
        ----------
        component_shapes: array
            transposed list of the model component vectors.

        Returns
        -------
        res: scipy.optimize.OptimizeResult
            return value of `scipy.optimize.minimize`
        """
        assert component_shapes.shape == (self.Ndata, self.Ncomponents), (
            component_shapes.shape,
            (self.Ndata, self.Ncomponents),
        )
        X = component_shapes
        y = self.flat_data
        offy = self.poisson_guess_data_offset
        offX = self.poisson_guess_model_offset
        x0 = np.log(
            np.median(
                (y.reshape((-1, 1)) + offy) / (X + offX),
                axis=0,
            )
        )
        res = minimize(
            poisson_negloglike, x0, args=(X, y),
            **self.minimize_kwargs)
        return res

    def loglike_poisson(self, component_shapes):
        """Return profile likelihood.

        Parameters
        ----------
        component_shapes: array
            transposed list of the model component vectors.

        Returns
        -------
        loglike: float
            log-likelihood
        """
        res = self.loglike_poisson_optimize(component_shapes)
        if not res.success:
            # give penalty when ill-defined
            return -1e100 + -res.fun
        return -res.fun

    def norms_poisson(self, component_shapes):
        """Return optimal normalisations.

        Parameters
        ----------
        component_shapes: array
            transposed list of the model component vectors.

        Returns
        -------
        norms: array
            normalisations, one value for each model component.
        """
        res = self.loglike_poisson_optimize(component_shapes)
        return exp(res.x)

    def sample_poisson(self, component_shapes, size, rng=np.random):
        """Sample from Poisson likelihood function.

        Sampling occurs with importance sampling,
        so the results need to be weighted by
        `exp(loglike_target-loglike_proposal)`
        or rejection sampled.


        Parameters
        ----------
        component_shapes: array
            transposed list of the model component vectors.
        size: int
            Maximum number of samples to generate.
        rng: object
            Random number generator

        Returns
        -------
        samples: array
            list of sampled normalisations. May be fewer than
            `size`, because negative normalisations are discarded.
        loglike_proposal: array
            for each sample, the importance sampling log-probability
        loglike_target: array
            for each sample, the Poisson log-likelihood
        """
        res = self.loglike_poisson_optimize(component_shapes)
        X = component_shapes
        profile_loglike = -res.fun
        # get mean
        counts = self.flat_data
        lognorms = res.x
        mean = exp(lognorms)
        lambda_hat = mean @ X.T
        D = np.diag(1 / lambda_hat)
        # Compute the Fisher Information Matrix
        FIM = X.T @ D @ X
        covariance = np.linalg.inv(FIM)
        samples_all = rng.multivariate_normal(mean, covariance, size=size)
        mask = np.all(samples_all > 0, axis=1)
        samples = samples_all[mask, :]
        # compute Poisson and Gaussian likelihood of these samples:
        rv = multivariate_normal(mean, covariance)
        # proposal probability: Gaussian
        loglike_proposal = rv.logpdf(samples) + profile_loglike
        lam = samples @ X.T
        # target probability function: Poisson
        loglike_target = np.sum(counts * log(lam) - lam, axis=1)
        return samples, loglike_proposal, loglike_target

    def loglike_gauss_optimize(self, component_shapes):
        """Optimize the normalisations assuming a Gaussian data model.

        Parameters
        ----------
        component_shapes: array
            transposed list of the model component vectors.

        Returns
        -------
        gauss_reg: LinearRegression
            Fitted scikit-learn regressor object.
        """
        X = component_shapes
        y = self.flat_data
        self.gauss_reg.fit(X, y, self.flat_invvar)
        return self.gauss_reg

    def loglike_gauss(self, component_shapes):
        """Return profile likelihood.

        Parameters
        ----------
        component_shapes: array
            transposed list of the model component vectors.

        Returns
        -------
        loglike: float
            log-likelihood
        """
        gauss_reg = self.loglike_gauss_optimize(component_shapes)
        X = component_shapes
        y = self.flat_data
        ypred = gauss_reg.predict(X)
        loglike = -0.5 * np.sum((ypred - y) ** 2 * self.flat_invvar)

        W = self.invvar_matrix
        XTWX = X.T @ W @ X
        cond = np.linalg.cond(XTWX)
        if cond > self.cond_threshold:
            penalty = -1e100 * (1 + self.cond_threshold)
        else:
            penalty = 0
        return loglike + penalty

    def norms_gauss(self, component_shapes):
        """Return optimal normalisations.

        Parameters
        ----------
        component_shapes: array
            transposed list of the model component vectors.

        Returns
        -------
        norms: array
            normalisations, one value for each model component.
        """
        gauss_reg = self.loglike_gauss_optimize(component_shapes)
        return gauss_reg.coef_

    def sample_gauss(self, component_shapes, size, rng=np.random):
        """Sample from Gaussian covariance matrix.

        Parameters
        ----------
        component_shapes: array
            transposed list of the model component vectors.
        size: int
            Number of samples to generate.
        rng: object
            Random number generator

        Returns
        -------
        samples: array
            list of sampled normalisations
        """
        gauss_reg = self.loglike_gauss_optimize(component_shapes)
        # get mean
        mean = gauss_reg.coef_
        # Compute covariance matrix
        X = component_shapes
        W = self.invvar_matrix
        XTWX = X.T @ W @ X
        covariance = np.linalg.inv(XTWX)
        return rng.multivariate_normal(mean, covariance, size=size)


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
    samples = statmodel.sample_gauss(X, 10000, rng)
    assert np.all(samples > 0)
    # plot 
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 6))
    plt.plot(x, model)
    plt.errorbar(x, data, noise, capsize=2, elinewidth=0.5, linestyle=' ')
    for sample in samples[::400]:
        plt.plot(x, X @ sample, ls='-', lw=1, alpha=0.5)
        np.testing.assert_allclose(sample, reg.coef_, atol=0.1, rtol=0.2)
    plt.savefig('testgausssampling.pdf')
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
    y = data
    def minfunc(lognorms):
        lam = np.exp(lognorms) @ X.T
        loglike = data * np.log(lam) - lam
        # print('  ', lognorms, loglike.sum())
        return -loglike.sum()

    x0 = np.log(np.median((data.reshape((-1, 1)) + 0.1) / (X + 0.1), axis=0))
    res = minimize(minfunc, x0, method='L-BFGS-B')
    norms_expected = np.exp(res.x)

    statmodel = ComponentModel(3, data)
    logl = statmodel.loglike_poisson(X)
    logl_expected = -poisson_negloglike(res.x, X, data)
    assert np.isclose(logl, logl_expected), (logl, logl_expected)
    norms_inferred = statmodel.norms_poisson(X)
    np.testing.assert_allclose(norms_inferred, [2.71413583, 0.46963565, 5.45321002])
    np.testing.assert_allclose(norms_inferred, norms_expected)
    samples, loglike_proposal, loglike_target = statmodel.sample_poisson(X, 100000, rng)
    assert np.all(samples > 0)
    Nsamples = len(samples)
    assert samples.shape == (Nsamples, 3), samples.shape
    assert loglike_proposal.shape == (Nsamples,)
    assert loglike_target.shape == (Nsamples,)
    # plot 
    import matplotlib.pyplot as plt
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
    y = data
    def minfunc(lognorms):
        lam = np.exp(lognorms) @ X.T
        loglike = data * np.log(lam) - lam
        # print('  ', lognorms, loglike.sum())
        return -loglike.sum()

    x0 = np.log(np.median((data.reshape((-1, 1)) + 0.1) / (X + 0.1), axis=0))
    res = minimize(minfunc, x0, method='L-BFGS-B')
    norms_expected = np.exp(res.x)

    statmodel = ComponentModel(3, data)
    logl = statmodel.loglike_poisson(X)
    logl_expected = -poisson_negloglike(res.x, X, data)
    assert np.isclose(logl, logl_expected), (logl, logl_expected)
    norms_inferred = statmodel.norms_poisson(X)
    np.testing.assert_allclose(norms_inferred, [29.94286524, 4.73544127, 51.15153849])
    np.testing.assert_allclose(norms_inferred, norms_expected)
    samples, loglike_proposal, loglike_target = statmodel.sample_poisson(X, 100000, rng)
    assert np.all(samples > 0)
    Nsamples = len(samples)
    assert samples.shape == (Nsamples, 3), samples.shape
    assert loglike_proposal.shape == (Nsamples,)
    assert loglike_target.shape == (Nsamples,)
    # plot 
    import matplotlib.pyplot as plt
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




