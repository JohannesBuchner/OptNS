"""Profile likelihoods."""
import jax
import numpy as np
from numpy import exp, log
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, norm

jax.config.update("jax_enable_x64", True)


def unique_components(X, tol=1e-12):
    """Identify components.

    Returns a boolean mask where each entry is True if the corresponding column
    in X is different from all previous columns.

    Parameters
    ----------
    X: array
        transposed list of the model component vectors.
    tol: float
        tolerance for comparing components

    Returns
    -------
    mask: array
        boolean mask of shape X.shape[1]
    """
    n_cols = X.shape[1]
    mask = np.ones(n_cols, dtype=bool)
    for i in range(1, n_cols):
        for j in range(i):
            if mask[j] and np.allclose(X[:, i], X[:, j], atol=tol, rtol=0):
                mask[i] = False
                break
    return mask


class GaussianPrior:
    """Gaussian prior probability function."""

    def __init__(self, means, cov):
        """Initialise.

        Parameters
        ----------
        means: array
            matrix of offsets between pairs of parameters
        cov: array
            matrix of covariance between pairs of parameters

        Attributes
        ----------
        indicator: array|None
            index array, one entry in each row should be -1, another 1.
        offsets: array|None
            mean offset of L == -1 and L == 1 entries
        Sigma_inv: array|None
            inverse covariance of L == -1 and L == 1 entries
        """
        ndim, = means.shape
        assert (ndim, ndim) == cov.shape
        self.mean = means
        self.cov_inv = np.linalg.inv(cov)
        self.norm_const = 0.5 * np.log(np.linalg.det(2 * np.pi * cov))

    def neglogprob(self, lognorms):
        """Compute negative log-probability.

        Parameters
        ----------
        lognorms: array
            logarithm of normalisations

        Returns
        -------
        neglogprob: float
            negative log-likelihood
        """
        delta = lognorms - self.mean
        return 0.5 * delta @ self.cov_inv @ delta + self.norm_const

    def logprob_many(self, lognorms):
        """Compute log-probability in a vectorized fashion.

        Parameters
        ----------
        lognorms: array
            2d array of logarithm of normalisations

        Returns
        -------
        neglogprob: float
            log-likelihood
        """
        delta = lognorms - self.mean.reshape((1, -1))
        return -0.5 * np.einsum('ni,ij,nj->n', delta, self.cov_inv, delta) - self.norm_const

    def grad(self, lognorms):
        """Compute gradient of neglogprob.

        Parameters
        ----------
        lognorms: array
            2d array of logarithm of normalisations

        Returns
        -------
        grad: array
            vector of gradients
        """
        delta = lognorms - self.mean
        return self.cov_inv @ delta

    def hessian(self, lognorms):
        """Compute Hessian matrix of log-prob w.r.t. lognorms.

        Parameters
        ----------
        lognorms: array
            2d array of logarithm of normalisations

        Returns
        -------
        hessian: array
            Hessian matrix
        """
        return self.cov_inv


@jax.jit
def poisson_negloglike(lognorms, X, counts, gaussprior=None):
    """Compute negative log-likelihood of a Poisson distribution.

    Parameters
    ----------
    lognorms: array
        logarithm of normalisations
    X: array
        transposed list of the model component vectors.
    counts: array
        non-negative integers giving the observed counts.
    gaussprior: GaussPrior
        Gaussian prior information

    Returns
    -------
    negloglike: float
        negative log-likelihood, neglecting the `1/fac(counts)` constant.
    """
    lam = jax.numpy.exp(lognorms) @ X.T
    loglike = counts * jax.numpy.log(lam) - lam
    negloglike = -loglike.sum()
    if gaussprior is not None:
        negloglike += gaussprior.neglogprob(lognorms)
    return negloglike


@jax.jit
def poisson_negloglike_grad(lognorms, X, counts, gaussprior=None):
    """Compute gradient of negative log-likelihood of a Poisson distribution.

    Parameters
    ----------
    lognorms: array
        logarithm of normalisations
    X: array
        transposed list of the model component vectors.
    counts: array
        non-negative integers giving the observed counts.
    gaussprior: GaussPrior
        Gaussian prior information

    Returns
    -------
    grad: array
        vector of gradients
    """
    norms = jax.numpy.exp(lognorms)
    lam = norms @ X.T
    diff = 1 - counts / lam
    grad = (diff @ X) * norms
    if gaussprior is not None:
        grad += gaussprior.grad(lognorms)
    return grad


def poisson_laplace_approximation(lognorms, X, gaussprior=None):
    """Compute mean and covariance corresponding to Poisson likelihood.

    Parameters
    ----------
    lognorms: array
        logarithm of normalisations
    X: array
        transposed list of the model component vectors.
    gaussprior: GaussPrior
        Gaussian prior information

    Returns
    -------
    mean: array
        peak of the log-likelihood, exp(lognorms)
    cov: array
        covariance of the Gaussian approximation to the log-likelihood,
        from the inverse Fisher matrix.
    """
    mean = exp(lognorms)
    lambda_hat = mean @ X.T
    D = np.diag(1 / lambda_hat)
    # Compute the Fisher Information Matrix
    FIM = X.T @ D @ X
    if gaussprior is not None:
        FIM += gaussprior.hessian(lognorms)
    covariance = np.linalg.inv(FIM)
    return mean, covariance


def gauss_importance_sample_stable(mean, covariance, size, rng):
    """Sample from a multivariate Gaussian.

    In case of numerical instability, only the diagonal of the covariance
    is used (mean-field approximation).

    Parameters
    ----------
    mean: array
        mean of the Gaussian
    covariance: array
        covariance matrix of the Gaussian.
    size: int
        Number of samples to generate.
    rng: object
        Random number generator

    Returns
    -------
    samples: array
        Generated samples
    logpdf: function
        logpdf of Gaussian proposal.
    """
    try:
        rv = multivariate_normal(mean, covariance)
        samples_all = rv.rvs(size=size, random_state=rng).reshape((size, len(mean)))
        rv_logpdf = rv.logpdf
    except np.linalg.LinAlgError:
        # fall back to diagonal approximation
        stdev = np.diag(covariance)**0.5
        rv = norm(mean, stdev)
        samples_all = rng.normal(mean, stdev, size=(size, len(mean))).reshape((size, len(mean)))

        def rv_logpdf(x):
            """Combine Gaussian logpdf of independent data.

            Parameters
            ----------
            x: array
                observations, 2d array

            Returns
            -------
            logprob: array
                1d array, summed over first axis.
            """
            return rv.logpdf(x).sum(axis=1)
    return samples_all, rv_logpdf


def poisson_initial_guess_heuristic(X, counts, epsilon_model, epsilon_data=0.1):
    """Guess component normalizations from counts.

    Based on the median count to model ratio of the components.

    Parameters
    ----------
    X: array
        transposed list of the model component vectors.
    counts: array
        non-negative integers giving the observed counts.
    epsilon_model: float
        small number to add to model components to avoid division by zero.
    epsilon_data: float
        small number to add to counts to avoid zeros.

    Returns
    -------
    lognorms: array
        logarithm of normalisations
    """
    counts_pos = counts.reshape((-1, 1)) + epsilon_data
    components_pos = X + epsilon_model
    N0 = np.median(counts_pos / components_pos, axis=0)
    return np.log(N0)


class PoissonModel:
    """Additive model components with Poisson measurements."""

    def __init__(self, Ncomponents, flat_data, positive=True, eps_model=0.1, eps_data=0.1, gaussprior=None):
        """Initialise model for Poisson data with additive model components.

        Parameters
        ----------
        flat_data: array
            Observed counts (non-negative integer numbers)
        positive: bool
            If true, only positive model components and normalisations are allowed.
        eps_model: float
            For heuristic initial guess of normalisations, small number to add to model component shapes.
        eps_data: float
            For heuristic initial guess of normalisations, small number to add to counts.
        gaussprior: GaussPrior
            Gaussian prior information
        """
        if not np.all(flat_data.astype(int) == flat_data) or not np.all(flat_data >= 0):
            raise AssertionError("Data are not counts, cannot use Poisson likelihood")
        self.Ncomponents = Ncomponents
        self.positive = positive
        self.guess_data_offset = eps_data
        self.guess_model_offset = eps_model
        self.minimize_kwargs = dict(method="L-BFGS-B", options=dict(ftol=1e-10, maxfun=10000))
        self.Ndata, = flat_data.shape
        self.flat_data = flat_data
        self.flat_invvar = None
        self.res = None
        self.gaussprior = gaussprior

    def update_components(self, component_shapes):
        """Set the model components.

        Parameters
        ----------
        component_shapes: array
            transposed list of the model component vectors.
        """
        self.mask_unique = unique_components(component_shapes)
        assert component_shapes.shape == (self.Ndata, self.Ncomponents)
        X = component_shapes
        if not np.isfinite(X).all():
            raise AssertionError("Component shapes are not all finite numbers.")
        if not np.any(np.abs(X) > 0, axis=1).all():
            raise AssertionError("In portions of the data set, all component shapes are zero. Problem is ill-defined.")
        if not np.any(np.abs(X) > 0, axis=0).all():
            raise AssertionError(f"Some components are exactly zero everywhere, so normalisation is ill-defined. Components: {np.any(X > 0, axis=0)}.")
        if self.positive and not np.all(X >= 0):
            raise AssertionError(f"Components must not be negative. Components: {~np.all(X >= 0, axis=0)}")
        self.X = X
        self._optimize()

    def _optimize(self):
        """Optimize the normalisations."""
        y = self.flat_data
        X = self.X
        mask_unique = self.mask_unique
        x0 = poisson_initial_guess_heuristic(
            self.X[:,self.mask_unique], y,
            self.guess_model_offset, self.guess_data_offset)
        # x0_rigorous = poisson_initial_guess(X[:,mask_unique], y, 0.1, 1e-50)
        assert np.isfinite(x0).all(), (x0, y, X, mask_unique)
        res = minimize(
            poisson_negloglike, x0, args=(X[:,mask_unique], y, self.gaussprior),
            jac=poisson_negloglike_grad,
            **self.minimize_kwargs)
        xfull = np.zeros(len(mask_unique)) + -1e50
        xfull[mask_unique] = res.x
        res.x = xfull
        self.res = res

    def loglike(self):
        """Get profile log-likelihood.

        Returns
        -------
        loglike: float
            log-likelihood value at optimized normalisations.
        """
        if self.res is None:
            raise AssertionError('need to call optimize() first!')
        if not self.res.success:
            # give penalty when ill-defined
            return -1e100
        return -self.res.fun

    def norms(self):
        """Get optimal component normalisations.

        Returns
        -------
        norms: array
            normalisations that optimize the likelihood for the current component shapes.
        """
        if self.res is None:
            raise AssertionError('need to call optimize() first!')
        return exp(self.res.x)

    def laplace_approximation(self):
        """Get Laplace approximation.

        Returns
        -------
        mean: array
            optimal component normalisations, same as norms()
        cov: array
            covariance matrix, from inverse Fisher matrix.
        """
        if self.res is None:
            raise AssertionError('need to call optimize() first!')
        return poisson_laplace_approximation(self.res.x, self.X, self.gaussprior)

    def sample(self, size, rng=np.random):
        """Sample from Laplace approximation to likelihood function.

        Parameters
        ----------
        size: int
            Maximum number of samples to generate.
        rng: object
            Random number generator

        Returns
        -------
        samples: array
            list of sampled normalisations. May be fewer than
            `size`, because negative normalisations are discarded
            if ComponentModel was initialized with positive=True.
        loglike_proposal: array
            for each sample, the importance sampling log-probability
        loglike_target: array
            for each sample, the Poisson log-likelihood
        """
        if self.res is None:
            raise AssertionError('need to call optimize() first!')
        res = self.res
        X = self.X
        profile_loglike = self.loglike()
        assert np.isfinite(profile_loglike), res
        # get mean
        counts = self.flat_data
        mean, covariance = self.laplace_approximation()
        samples_all, rv_logpdf = gauss_importance_sample_stable(mean, covariance, size, rng=rng)

        if self.positive:
            mask = np.all(samples_all > 0, axis=1)
            samples = samples_all[mask, :]
        else:
            samples = samples_all
        # compute Poisson and Gaussian likelihood of these samples:
        # proposal probability: Gaussian
        loglike_gauss_proposal = rv_logpdf(samples) - rv_logpdf(mean.reshape((1, -1)))
        assert np.isfinite(loglike_gauss_proposal).all(), (
            samples[~np.isfinite(loglike_gauss_proposal),:], loglike_gauss_proposal[~np.isfinite(loglike_gauss_proposal)])
        loglike_proposal = loglike_gauss_proposal + profile_loglike
        assert np.isfinite(loglike_proposal).all(), (samples, loglike_proposal, loglike_gauss_proposal)
        lam = samples @ X.T
        # target probability function: Poisson
        loglike_target = np.sum(counts * log(lam) - lam, axis=1)
        if self.gaussprior is not None:
            loglike_target += self.gaussprior.logprob_many(np.log(samples))
        return samples, loglike_proposal, loglike_target


class GaussModel:
    """Additive model components with Gaussian measurements."""

    def __init__(self, Ncomponents, flat_data, flat_invvar, positive, cond_threshold=1e6):
        """Initialise model for Gaussian data with additive model components.

        Parameters
        ----------
        flat_data: array
            Measurement errors (non-negative integer numbers)
        flat_invvar: array
            Inverse Variance of measurement errors (yerr**-2). Must be non-negative
        positive: bool
            If true, only positive model components and normalisations are allowed.
        cond_threshold: float
            Threshold for numerical stability condition (see `np.linalg.cond`).
        """
        if not np.isfinite(flat_data).all():
            raise AssertionError("Invalid data, not finite numbers.")
        self.Ncomponents = Ncomponents
        self.Ndata, = flat_data.shape
        assert (self.Ndata,) == flat_invvar.shape, (self.Ndata, flat_invvar.shape)
        self.positive = positive
        self.cond_threshold = cond_threshold
        self.flat_data = flat_data
        self.update_noise(flat_invvar)
        self.res = None

    def update_noise(self, flat_invvar):
        """Set the measurement error.

        Parameters
        ----------
        flat_invvar: array
            Inverse Variance of measurement errors (yerr**-2). Must be non-negative
        """
        if not (flat_invvar > 0).all():
            raise AssertionError("Inverse variance must be positive")
        self.flat_invvar = flat_invvar
        self.W = np.sqrt(flat_invvar)
        self.invvar_matrix = np.diag(self.flat_invvar)
        # 1 / sqrt(2 pi sigma^2) term:
        self.loglike_prefactor = 0.5 * np.sum(np.log(self.flat_invvar / (2 * np.pi)))

    def update_components(self, component_shapes):
        """Set the model components.

        Parameters
        ----------
        component_shapes: array
            transposed list of the model component vectors.
        """
        assert component_shapes.ndim == 2
        X = component_shapes
        assert component_shapes.shape == (self.Ndata, self.Ncomponents)
        if not np.isfinite(X).all():
            raise AssertionError("Component shapes are not all finite numbers.")
        if not np.any(np.abs(X) > 0, axis=1).all():
            raise AssertionError("In portions of the data set, all component shapes are zero. Problem is ill-defined.")
        if not np.any(np.abs(X) > 0, axis=0).all():
            raise AssertionError(f"Some components are exactly zero everywhere, so normalisation is ill-defined. Components: {np.any(X > 0, axis=0)}.")
        if self.positive and not np.all(X >= 0):
            raise AssertionError(f"Components must not be negative. Components: {~np.all(X >= 0, axis=0)}")
        self.X = X
        self.Xw = X * self.W[:, None]
        self.yw = self.flat_data * self.W
        self.XT_X = self.Xw.T @ self.Xw
        self.XT_y = self.Xw.T @ self.yw

        # self.W = self.invvar_matrix
        # self.XTWX = X.T @ W @ X
        self.cond = np.linalg.cond(self.XT_X)
        self._optimize()

    def _optimize(self):
        """Optimize the normalisations."""
        if self.cond > self.cond_threshold:
            self.res = np.linalg.pinv(self.XT_X, rcond=self.cond_threshold) @ self.XT_y
        else:
            self.res = np.linalg.solve(self.XT_X, self.XT_y)

    def loglike(self):
        """Return profile likelihood.

        Returns
        -------
        loglike: float
            log-likelihood
        """
        if self.res is None:
            raise AssertionError('need to call optimize() first!')
        loglike_plain = -0.5 * self.chi2() + self.loglike_prefactor

        if self.cond > self.cond_threshold:
            penalty = -1e100 * (1 + self.cond)
        else:
            penalty = 0
        return loglike_plain + penalty

    def chi2(self):
        """Return chi-square.

        Returns
        -------
        chi2: float
            Inverse variance weighted sum of squared deviations.
        """
        if self.res is None:
            raise AssertionError('need to call optimize() first!')
        ypred = np.dot(self.X, self.res)
        return np.sum((ypred - self.flat_data) ** 2 * self.flat_invvar)

    def norms(self):
        """Return optimal normalisations.

        Normalisations of subsequent identical components will be zero.

        Returns
        -------
        norms: array
            normalisations, one value for each model component.
        """
        if self.res is None:
            raise AssertionError('need to call optimize() first!')
        return self.res

    def sample(self, size, rng=np.random):
        """Sample from Gaussian covariance matrix.

        Parameters
        ----------
        size: int
            Number of samples to generate.
        rng: object
            Random number generator

        Returns
        -------
        samples: array
            list of sampled normalisations
        loglike_proposal: array
            likelihood for sampled points
        loglike_target: array
            likelihood of optimized point used for importance sampling
        """
        if self.res is None:
            raise AssertionError('need to call optimize() first!')
        mean = self.res
        loglike_profile = self.loglike()
        # Compute covariance matrix
        X = self.X
        covariance = np.linalg.inv(self.XT_X)
        samples_all = rng.multivariate_normal(mean, covariance, size=size)

        rv = multivariate_normal(mean, covariance)

        if self.positive:
            mask = np.all(samples_all > 0, axis=1)
            samples = samples_all[mask, :]
        else:
            samples = samples_all

        y_pred = samples @ X.T
        loglike_gauss_proposal = rv.logpdf(samples) - rv.logpdf(mean)
        loglike_target = -0.5 * np.sum(
            (y_pred - self.flat_data)**2 * self.flat_invvar,
            axis=1,
        ) + self.loglike_prefactor
        loglike_proposal = loglike_profile + loglike_gauss_proposal

        return samples, loglike_proposal, loglike_target


class GPModel:
    """Additive model components with Gaussian Process correlated measurements."""

    def __init__(self,Ncomponents, flat_data, gp, positive, cond_threshold=1e6):
        """Initialise model for Gaussian data with additive model components.

        Parameters
        ----------
        flat_data: array
            Measurement errors (non-negative integer numbers)
        gp: object
            Gaussian process object from george or celerite
        positive: bool
            If true, only positive model components and normalisations are allowed.
        cond_threshold: float
            Threshold for numerical stability condition (see `np.linalg.cond`).
        """
        if not np.isfinite(flat_data).all():
            raise AssertionError("Invalid data, not finite numbers.")
        self.Ncomponents = Ncomponents
        self.Ndata, = flat_data.shape
        self.positive = positive
        self.cond_threshold = cond_threshold
        self.flat_data = flat_data
        self.gp = gp
        self.res = None

    def update_components(self, component_shapes):
        """Set the model components.

        Parameters
        ----------
        component_shapes: array
            transposed list of the model component vectors.
        """
        assert component_shapes.ndim == 2
        X = component_shapes
        assert component_shapes.shape == (self.Ndata, self.Ncomponents)
        if not np.isfinite(X).all():
            raise AssertionError("Component shapes are not all finite numbers.")
        if not np.any(np.abs(X) > 0, axis=1).all():
            raise AssertionError("In portions of the data set, all component shapes are zero. Problem is ill-defined.")
        if not np.any(np.abs(X) > 0, axis=0).all():
            raise AssertionError(f"Some components are exactly zero everywhere, so normalisation is ill-defined. Components: {np.any(X > 0, axis=0)}.")
        if self.positive and not np.all(X >= 0):
            raise AssertionError(f"Components must not be negative. Components: {~np.all(X >= 0, axis=0)}")
        self.X = X
        self.Kinv_X = self.gp.apply_inverse(X)
        self.Kinv_y = self.gp.apply_inverse(self.flat_data)
        self.XTKinvX = self.X.T @ self.Kinv_X
        self.XTKinvy = self.X.T @ self.Kinv_y

        self.cond = np.linalg.cond(self.XTKinvX)
        self._optimize()

    def _optimize(self):
        """Optimize the normalisations."""
        if self.cond > self.cond_threshold:
            self.res = np.linalg.pinv(self.XTKinvX, rcond=self.cond_threshold) @ self.XTKinvy
        else:
            self.res = np.linalg.solve(self.XTKinvX, self.XTKinvy)

    def loglike(self):
        """Return profile likelihood.

        Returns
        -------
        loglike: float
            log-likelihood
        """
        if self.res is None:
            raise AssertionError('need to call optimize() first!')
        y_pred = self.res @ self.X.T
        loglike_plain = self.gp.log_likelihood(self.flat_data - y_pred) + self.gp.log_prior()

        if self.cond > self.cond_threshold:
            penalty = -1e100 * (1 + self.cond)
        else:
            penalty = 0
        return loglike_plain + penalty

    def norms(self):
        """Return optimal normalisations.

        Normalisations of subsequent identical components will be zero.

        Returns
        -------
        norms: array
            normalisations, one value for each model component.
        """
        if self.res is None:
            raise AssertionError('need to call optimize() first!')
        return self.res

    def sample(self, size, rng=np.random):
        """Sample from Gaussian covariance matrix.

        Parameters
        ----------
        size: int
            Number of samples to generate.
        rng: object
            Random number generator

        Returns
        -------
        samples: array
            list of sampled normalisations
        loglike_proposal: array
            likelihood for sampled points
        loglike_target: array
            likelihood of optimized point used for importance sampling
        """
        if self.res is None:
            raise AssertionError('need to call optimize() first!')
        mean = self.res
        loglike_profile = self.loglike()
        covariance = np.linalg.inv(self.XTKinvX)
        samples_all = rng.multivariate_normal(mean, covariance, size=size)

        rv = multivariate_normal(mean, covariance)

        if self.positive:
            mask = np.all(samples_all > 0, axis=1)
            samples = samples_all[mask, :]
        else:
            samples = samples_all

        y_preds = samples @ self.X.T
        loglike_gauss_proposal = rv.logpdf(samples) - rv.logpdf(mean)
        loglike_target = np.array([
            self.gp.log_likelihood(self.flat_data - y_pred) for y_pred in y_preds]) + self.gp.log_prior()
        loglike_proposal = loglike_profile + loglike_gauss_proposal

        return samples, loglike_proposal, loglike_target


def ComponentModel(Ncomponents, flat_data, flat_invvar=None, positive=True, **kwargs):
    """Generalized Additive Model.

    Defines likelihoods for observed data,
    given arbitrary components which are
    linearly added with non-negative normalisations.

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
    positive: bool
        whether Gaussian normalisations must be positive.
    **kwargs: dict
        additional arguments passed to `PoissonModel` (if flat_invvar is None)
        or `GaussModel` (otherwise)

    Returns
    -------
    model: object
        `PoissonModel` if flat_invvar is None or otherwise `GaussModel`
    """
    if flat_invvar is None:
        return PoissonModel(Ncomponents, flat_data, positive=positive, **kwargs)
    else:
        return GaussModel(Ncomponents, flat_data, flat_invvar, positive=positive, **kwargs)
