"""Prior probability functions."""
import numpy as np


class GaussianPrior:
    """Gaussian prior probability function."""

    def __init__(self, mean, stdevs):
        """Initialise.

        Parameters
        ----------
        mean: array
            mean
        stdevs: array
            standard deviations

        Attributes
        ----------
        norm_const: float
            normalisation for logprob function
        cov_inv: array
            inverse covariance matrix
        """
        ndim, = mean.shape
        self.mean = mean
        self.stdevs = stdevs
        self.inv_cov = np.diag(self.stdevs**-2)

    def __str__(self):
        """Make a string representation."""
        return f'GaussianPrior({self.mean}, {self.stdevs})'

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
        return 0.5 * (((lognorms - self.mean) / self.stdevs)**2).sum()

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
        return -0.5 * (((lognorms - self.mean) / self.stdevs)**2).sum(axis=1)

    def grad(self, lognorms):
        """Compute gradient of neglogprob.

        Parameters
        ----------
        lognorms: array
            logarithm of normalisations

        Returns
        -------
        grad: array
            vector of gradients
        """
        return (lognorms - self.mean) / self.stdevs**2

    def hessian(self, lognorms):
        """Compute Hessian matrix of log-prob w.r.t. lognorms.

        Parameters
        ----------
        lognorms: array
            logarithm of normalisations

        Returns
        -------
        hessian: array
            Hessian matrix
        """
        return self.inv_cov


class SimilarityPrior:
    """Prior probability function making parameters similar."""

    def __init__(self, participants, std):
        """Initialise.

        Parameters
        ----------
        participants: array
            boolean mask of participating parameters
        std: float
            standard deviation on deviations across participating parameters
        """
        assert participants.dtype == bool, participants.dtype
        ndim, = participants.shape
        self.df = participants.sum() - 1
        n = participants.sum()
        self.participants = participants
        self.std = float(std)
        ones = np.ones((n, n)) / n
        self.H = (np.eye(n) - ones) / std**2

    def __str__(self):
        """Make a string representation."""
        return f'SimilarityPrior({self.participants * 1}, {self.std})'

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
        vals = lognorms[self.participants]
        chi2_value = np.sum((vals - np.mean(vals))**2) / self.std**2
        return 0.5 * chi2_value

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
        vals = lognorms[:,self.participants]
        chi2_value = np.sum((vals - np.mean(vals))**2, axis=1) / self.std**2
        return -0.5 * chi2_value

    def grad(self, lognorms):
        """Compute gradient of neglogprob.

        Parameters
        ----------
        lognorms: array
            logarithm of normalisations

        Returns
        -------
        grad: array
            vector of gradients
        """
        vals = lognorms[self.participants]
        grad = np.zeros_like(lognorms)
        grad[self.participants] = (vals - np.mean(vals)) / self.std**2
        return grad

    def hessian(self, lognorms):
        """Compute Hessian matrix of log-prob w.r.t. lognorms.

        Parameters
        ----------
        lognorms: array
            logarithm of normalisations (ignored)

        Returns
        -------
        hessian: array
            Hessian matrix
        """
        return self.H
