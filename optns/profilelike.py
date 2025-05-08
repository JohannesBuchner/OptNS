"""Profile likelihoods."""
import jax
import numpy as np
from jax import jit
from numpy import exp, log
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, norm

jax.config.update("jax_enable_x64", True)
# def jit(func): return func


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
        # mask[i] = (X[:,i] == 0).any()
        for j in range(i):
            if mask[j] and np.allclose(X[:, i], X[:, j], atol=tol, rtol=0):
                mask[i] = False
                break
    return mask


@jit
def _poisson_negloglike(lognorms, X, counts, eps=1e-50):
    """Compute negative log-likelihood of a Poisson distribution.

    Parameters
    ----------
    lognorms: array
        logarithm of normalisations
    X: array
        transposed list of the model component vectors.
    counts: array
        non-negative integers giving the observed counts.
    eps: float
        small value to add to model component to avoid log(0)

    Returns
    -------
    negloglike: float
        negative log-likelihood, neglecting the `1/fac(counts)` constant.
    """
    lam = jax.numpy.exp(lognorms) @ X.T
    loglike = counts * jax.numpy.log(lam + eps) - lam
    return -loglike.sum()


@jit
def _poisson_negloglike_grad(lognorms, X, counts, eps=1e-50):
    """Compute gradient of negative log-likelihood of a Poisson distribution.

    Parameters
    ----------
    lognorms: array
        logarithm of normalisations
    X: array
        transposed list of the model component vectors.
    counts: array
        non-negative integers giving the observed counts.
    eps: float
        small value to add to model component to avoid log(0)

    Returns
    -------
    grad: array
        vector of gradients
    """
    norms = jax.numpy.exp(lognorms)
    lam = norms @ X.T
    diff = 1 - counts / (lam + eps)
    return (diff @ X) * norms


@jit
def _poisson_negloglike_hessian(lognorms, X, counts, eps=1e-50):
    """Compute Hessian of negative log-likelihood of a Poisson distribution.

    Parameters
    ----------
    lognorms: array
        logarithm of normalisations
    X: array
        transposed list of the model component vectors.
    counts: array
        non-negative integers giving the observed counts.
    eps: float
        small value to add to model component to avoid log(0)

    Returns
    -------
    grad: array
        vector of gradients
    """
    norms = jax.numpy.exp(lognorms)
    lam = norms @ X.T
    W = (lam + eps - counts) / ((lam + eps)**2)

    H = (X.T * W[None, :]) @ X
    H *= norms[:, None] * norms[None, :]
    assert H.shape == (len(lognorms), len(lognorms))
    return H


def poisson_negloglike(lognorms, X, counts, priors=[]):
    """Compute negative log-likelihood of a Poisson distribution.

    Parameters
    ----------
    lognorms: array
        logarithm of normalisations
    X: array
        transposed list of the model component vectors.
    counts: array
        non-negative integers giving the observed counts.
    priors: list
        List of prior objects

    Returns
    -------
    negloglike: float
        negative log-likelihood, neglecting the `1/fac(counts)` constant.
    """
    negloglike = _poisson_negloglike(lognorms, X, counts)
    for prior in priors:
        negloglike += prior.neglogprob(lognorms)
    return negloglike


def poisson_negloglike_grad(lognorms, X, counts, priors=[]):
    """Compute gradient of negative log-likelihood of a Poisson distribution.

    Parameters
    ----------
    lognorms: array
        logarithm of normalisations
    X: array
        transposed list of the model component vectors.
    counts: array
        non-negative integers giving the observed counts.
    priors: list
        List of prior objects

    Returns
    -------
    grad: array
        vector of gradients
    """
    grad = _poisson_negloglike_grad(lognorms, X, counts)
    for prior in priors:
        grad += prior.grad(lognorms)
    return grad


def poisson_negloglike_hessian(lognorms, X, counts, priors=[]):
    """Compute Hessian of negative log-likelihood of a Poisson distribution.

    Parameters
    ----------
    lognorms: array
        logarithm of normalisations
    X: array
        transposed list of the model component vectors.
    counts: array
        non-negative integers giving the observed counts.
    priors: list
        List of prior objects

    Returns
    -------
    grad: array
        vector of gradients
    """
    H = _poisson_negloglike_hessian(lognorms, X, counts)
    for prior in priors:
        H += prior.hessian(lognorms)
    return H


def poisson_laplace_approximation(lognorms, X, counts, priors=[], eps=1e-6):
    """Compute mean and covariance corresponding to Poisson likelihood.

    Parameters
    ----------
    lognorms: array
        logarithm of normalisations
    X: array
        transposed list of the model component vectors.
    counts: array
        non-negative integers giving the observed counts.
    priors: list
        List of prior objects
    eps: float
        small number to add to hessian before inverting it

    Returns
    -------
    mean: array
        peak of the log-likelihood, exp(lognorms)
    cov: array
        covariance of the Gaussian approximation to the log-likelihood,
        from the inverse Hessian matrix.
    """
    mean = exp(lognorms)
    lambda_hat = mean @ X.T
    D = np.diag(1 / lambda_hat)
    # Compute the Fisher Information Matrix
    FIM = X.T @ D @ X
    for prior in priors:
        FIM += prior.hessian(lognorms)
    covariance = np.linalg.inv(FIM)
    """
    mean = jax.numpy.exp(lognorms)
    hessian = poisson_negloglike_hessian(lognorms, X, counts, gaussprior)
    covariance = np.linalg.inv(hessian + eps * np.eye(len(hessian)))
    """
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


def poisson_initial_guess_heuristic(X, counts, connection_graph, epsilon_model, epsilon_data=0.1):
    """Guess component normalizations from counts.

    Based on matching the observed counts to each model component,
    weighted towards the regions where the model component is high.

    Parameters
    ----------
    X: array
        transposed list of the model component vectors.
    counts: array
        non-negative integers giving the observed counts.
    connection_graph: list
        one entry for each component, giving the mask of correlated components.
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
    assert (components_pos >= 0).all(), (X.min(), epsilon_model)
    assert X.shape[1] == len(connection_graph)
    N0 = np.zeros(len(connection_graph)) - 1
    for i, rows in enumerate(connection_graph):
        if N0[i] <= 0:
            # ratio between model and observation should be met by
            # N0[rows] = np.median(counts_pos / components_pos[:,rows])
            # weigh by shape
            N0[rows] = np.average(counts_pos / components_pos[:,rows], weights=components_pos[:,rows])
            # N0[rows] = np.mean(counts_pos / components_pos[:,rows])
            assert np.all(N0[rows] > 0), (N0[rows], counts_pos / components_pos[:,rows], counts_pos, components_pos[:,rows])
        assert np.all(N0[:i + 1] > 0), (i, rows, N0, connection_graph)
    return np.log(N0)


class PoissonModel:
    """Generalized Additive Model with Poisson measurements.

    Defines likelihoods for observed data,
    given arbitrary components which are
    linearly added with normalisations. if positive=True,
    normalisations are forced to be positive.
    """

    def __init__(self, Ncomponents, flat_data, positive=True, eps_model=0.1, eps_data=0.1, priors=[]):
        """Initialise model for Poisson data with additive model components.

        Parameters
        ----------
        Ncomponents: int
            number of model shape components
        flat_data: array
            Observed counts (non-negative integer numbers)
        positive: bool
            If true, only positive model components and normalisations are allowed.
        eps_model: float
            For heuristic initial guess of normalisations, small number to add to model component shapes.
        eps_data: float
            For heuristic initial guess of normalisations, small number to add to counts.
        priors: list
            List of prior objects
        """
        if not np.all(flat_data.astype(int) == flat_data) or not np.all(flat_data >= 0):
            raise AssertionError("Data are not counts, cannot use Poisson likelihood")
        self.Ncomponents = Ncomponents
        self.positive = positive
        self.guess_data_offset = eps_data
        self.guess_model_offset = eps_model
        self.minimize_kwargs = dict(method="L-BFGS-B", options=dict(ftol=1e-10, maxfun=10000))
        # self.minimize_kwargs = dict(method="Nelder-Mead", options=dict(fatol=1e-10, maxfev=10000))
        # you would think that the hessian helps
        # But actually it is slow. So commented out for now.
        # self.minimize_kwargs = dict(method="trust-ncg", options=dict(gtol=1e-10), hess=poisson_negloglike_hessian)
        self.Ndata, = flat_data.shape
        self.flat_data = flat_data
        self.flat_invvar = None
        self.res = None
        self.priors = priors
        self.mask_unique_all = np.ones(self.Ncomponents, dtype=bool)
        self.mask_unique = self.mask_unique_all

        self.connection_graph = []
        for i in range(self.Ncomponents):
            members = np.zeros(self.Ncomponents, dtype=bool)
            members[i] = True
            for prior in self.priors:
                if hasattr(prior, 'participants'):
                    if prior.participants[i]:
                        members[prior.participants] = True
            self.connection_graph.append(members)

    def update_components(self, component_shapes):
        """Set the model components.

        Parameters
        ----------
        component_shapes: array
            transposed list of the model component vectors.
        """
        if self.priors == []:
            self.mask_unique = unique_components(component_shapes)
        assert component_shapes.shape == (self.Ndata, self.Ncomponents)
        X = component_shapes
        if not np.isfinite(X).all():
            raise AssertionError("Component shapes are not all finite numbers.")
        if not np.any(np.abs(X) > 0, axis=1).all():
            raise AssertionError("In portions of the data set, all component shapes are zero. Problem is ill-defined.")
        for members in self.connection_graph:
            if not np.any(np.abs(X[:,members]) > 0):
                raise AssertionError(f"Component(s) {np.where(members)[0]} are zero everywhere, so normalisation is ill-defined.")
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
            [g[self.mask_unique] for g, m in zip(self.connection_graph, self.mask_unique) if m],
            self.guess_model_offset, self.guess_data_offset,
        )
        assert np.isfinite(x0).all(), (x0, y, X, mask_unique)
        res = minimize(
            poisson_negloglike, x0, args=(X[:,mask_unique], y, self.priors),
            jac=poisson_negloglike_grad,
            **self.minimize_kwargs)
        if not mask_unique.all():
            xfull = np.zeros(len(mask_unique)) + -1e50
            xfull[mask_unique] = res.x
            res.x = xfull
        if not res.success:
            print(res)
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
        return poisson_laplace_approximation(self.res.x, self.X, self.flat_data, self.priors)

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
        for prior in self.priors:
            loglike_target += prior.logprob_many(np.log(samples))
        return samples, loglike_proposal, loglike_target


class GaussModel:
    """Generalized Additive Model with independent Gaussian measurements.

    Defines likelihoods for observed data,
    given arbitrary components which are
    linearly added with normalisations. if positive=True,
    normalisations are forced to be positive.
    """

    def __init__(self, Ncomponents, flat_data, flat_invvar, positive, cond_threshold=1e6):
        """Initialise model for Gaussian data with additive model components.

        Parameters
        ----------
        Ncomponents: int
            number of model shape components
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
    """Generalized Additive Model with Gaussian Process correlated measurements.

    Defines likelihoods for observed data,
    given arbitrary components which are
    linearly added with normalisations. if positive=True,
    normalisations are forced to be positive.
    """

    def __init__(self, Ncomponents, flat_data, gp, positive, cond_threshold=1e6):
        """Initialise model for Gaussian data with additive model components.

        Parameters
        ----------
        Ncomponents: int
            number of model shape components
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
        assert component_shapes.shape == (self.Ndata, self.Ncomponents), (component_shapes.shape, (self.Ndata, self.Ncomponents))
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
