import numpy as np
import tqdm
from numpy import log
from ultranest import ReactiveNestedSampler

from profilelike import ComponentModel


def ess(w):
    """Compute the effective sample size.

    Parameters
    ----------
    w: array
        Weights.

    Returns
    -------
    ESS: float
        effective sample size.
    """
    return len(w) / (1.0 + ((len(w) * w - 1) ** 2).sum() / len(w))


class OptNS:
    """Optimized Nested Sampling.

    Defines likelihoods for observed data,
    given arbitrary components which are
    linearly added with non-negative normalisations.
    """

    def __init__(
        self,
        linear_param_names,
        nonlinear_param_names,
        compute_model_components,
        nonlinear_param_transform,
        linear_param_logprior,
        flat_data,
        flat_invvar=None,
        positive=True,
    ):
        """Initialise.

        Parameters
        ----------
        linear_param_names: list
            Names of the normalisation parameters.
        nonlinear_param_names: list
            Names of the non-linear parameters.
        compute_model_components: func
            function which computes a transposed list of model components,
            given the non-linear parameters.
        nonlinear_param_transform: func
            Prior probability transform function for the non-linear parameters.
        linear_param_logprior: func
            Prior log-probability function for the linear parameters.
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
        """
        Ncomponents = len(linear_param_names)
        self.linear_param_names = linear_param_names
        self.nonlinear_param_names = nonlinear_param_names
        self.nonlinear_param_transform = nonlinear_param_transform
        self.linear_param_logprior = linear_param_logprior
        self.compute_model_components = compute_model_components
        self.statmodel = ComponentModel(Ncomponents, flat_data, flat_invvar)

    def optlinearsample(self, nonlinear_params, size):
        """Sample linear parameters conditional on non-linear parameters.

        Parameters
        ----------
        nonlinear_params: array
            values of the non-linear parameters.
        size: int
            Maximum number of samples to return.

        Returns
        -------
        y_pred: array
            Predicted model for each sample. shape: (Nsamples, Ndata)
        params: array
            Posterior samples parameter vectors. shape: (Nsamples, Nlinear + Nnonlinear)
        logweights: array
            Log of importance sampling weights of the posterior samples. shape: (Nsamples,)
        """
        X = self.compute_model_components(nonlinear_params)
        if self.statmodel.flat_invvar is None:
            linear_params, loglike_proposal, loglike_target = (
                self.statmodel.sample_poisson(X, size)
            )
            logl_profile = loglike_target - loglike_proposal
        else:
            linear_params = self.statmodel.sample_gauss(X, size)
            logl_profile = self.statmodel.loglike_gauss(X)
        Nsamples, Nlinear = linear_params.shape
        y_pred = linear_params @ X.T
        if self.statmodel.flat_invvar is None:
            logl_full = np.sum(self.statmodel.flat_data * log(y_pred) - y_pred, axis=1)
        else:
            logl_full = -0.5 * np.sum(
                ((y_pred - self.statmodel.flat_data) * self.statmodel.flat_invvar) ** 2,
                axis=1,
            )
        logprior = self.linear_param_logprior(linear_params)
        params = np.empty((Nsamples, len(nonlinear_params) + Nlinear))
        params[:, :Nlinear] = linear_params
        params[:, Nlinear:] = nonlinear_params.reshape((1, -1))
        return y_pred, params, logl_full + logprior - logl_profile - np.log(Nsamples)

    def loglikelihood(self, nonlinear_params):
        """Compute optimized log-likelihood function.

        Parameters
        ----------
        nonlinear_params: array
            values of the non-linear parameters.

        Returns
        -------
        loglike: float
            log-likelihood
        """
        X = self.compute_model_components(nonlinear_params)
        assert np.isfinite(X).all(), X
        if self.statmodel.flat_invvar is None:
            return self.statmodel.loglike_poisson(X)
        else:
            return self.statmodel.loglike_gauss(X)

    def ReactiveNestedSampler(self, **sampler_kwargs):
        """Create a nested sampler.

        Parameters
        ----------
        **sampler_kwargs: dict
            arguments passed to ReactiveNestedSampler.

        Returns
        -------
        sampler: ReactiveNestedSampler
            UltraNest sampler object.
        """
        self.sampler = ReactiveNestedSampler(
            self.nonlinear_param_names,
            self.loglikelihood,
            transform=self.nonlinear_param_transform,
            **sampler_kwargs,
        )
        return self.sampler

    def get_weighted_samples(self, oversample_factor):
        """Sample from full posterior.

        Parameters
        ----------
        oversample_factor: int
            Maximum number of conditional posterior samples on the
            linear parameters for each posterior sample
            from the nested sampling run with the non-linear parameters
            left to vary.

        Returns
        -------
        fullsamples: array
            Posterior samples parameter vectors. shape: (Nsamples, Nlinear + Nnonlinear)
        weights: array
            Importance sampling weights of the posterior samples. shape: (Nsamples,)
        y_pred: array
            Predicted model for each sample. shape: (Nsamples, Ndata)
        """
        optsamples = self.sampler.results["samples"]
        Noptsamples = len(optsamples)
        Nmaxsamples = Noptsamples * oversample_factor
        # go through posterior samples and sample normalisations
        Nsampled = 0
        logweights = np.empty(Nmaxsamples)
        y_preds = np.empty((Nmaxsamples, self.statmodel.Ndata))
        fullsamples = np.empty(
            (
                Nmaxsamples,
                len(self.linear_param_names) + len(self.nonlinear_param_names),
            )
        )
        for i, nonlinear_params in enumerate(tqdm.tqdm(optsamples)):
            y_pred_i, fullsamples_i, logweights_i = self.optlinearsample(
                nonlinear_params, size=oversample_factor
            )
            Nsampled_i = len(logweights_i)
            jlo = Nsampled
            jhi = Nsampled + Nsampled_i
            y_preds[jlo:jhi, :] = y_pred_i
            fullsamples[jlo:jhi, :] = fullsamples_i
            logweights[jlo:jhi] = logweights_i
            Nsampled += Nsampled_i

        y_preds = y_preds[:Nsampled, :]
        fullsamples = fullsamples[:Nsampled, :]
        logweights = logweights[:Nsampled]

        weights = np.exp(logweights - logweights.max())
        weights /= weights.sum()
        return fullsamples, weights, y_preds

    def resample(self, fullsamples, weights, y_preds, rng=np.random):
        """Resample weighted posterior samples into equally weighted samples.

        The number of returned samples depends on the effective sample
        size, as determined from the *weights*.

        Parameters
        ----------
        fullsamples: array
            Posterior samples parameter vectors. shape: (Nsamples, Nlinear + Nnonlinear)
        weights: array
            Importance sampling weights of the posterior samples. shape: (Nsamples,)
        y_preds: array
            Predicted model for each sample. shape: (Nsamples, Ndata)
        rng: object
            Random number generator.

        Returns
        -------
        fullsamples: array
            Posterior samples parameter vectors. shape: (ESS, Nlinear + Nnonlinear)
        y_preds: array
            Predicted model for each sample. shape: (ESS, Ndata)
        """
        rejection_sampled_indices = rng.choice(
            len(weights), p=weights, size=int(ess(weights))
        )
        return fullsamples[rejection_sampled_indices, :], y_preds[
            rejection_sampled_indices, :
        ]
