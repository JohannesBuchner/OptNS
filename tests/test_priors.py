import numpy as np
from optns.priors import GaussianPrior, SimilarityPrior
import scipy.stats
from numpy.testing import assert_allclose


def test_gaussian_prior():
    means = np.array([1.23, 4.56])
    stdevs = np.array([1, 0.1])
    gauss = GaussianPrior(means, stdevs)
    print(gauss)
    rva = scipy.stats.norm(1.23, 1)
    rvb = scipy.stats.norm(4.56, 0.1)
    logpdfa = rva.logpdf(2.0) - rva.logpdf(rva.mean())
    logpdfb = rvb.logpdf(4.0) - rvb.logpdf(rvb.mean())
    sample = np.array([2.0, 4.0])
    logpdf = -gauss.neglogprob(sample)
    print(logpdfa, logpdfb, logpdf)
    assert_allclose(logpdf, logpdfa + logpdfb)
    assert_allclose(logpdf, gauss.logprob_many(np.array([[2.0, 4.0]]))[0])
    assert_allclose(gauss.grad(sample), np.diag(stdevs**-2) @ (sample - means))
    assert_allclose(gauss.hessian(sample), np.diag(stdevs**-2))


def test_similarity_prior():
    participants = np.array([False, True, True])
    prior = SimilarityPrior(participants, 0.01)
    print(prior)
    assert prior.neglogprob(np.array([1.23, 4.56, 4.56])) == 0
    lp = -prior.neglogprob(np.array([1.23, 4.56, 4.55]))
    assert_allclose(lp, -0.5 * 0.5)
    assert_allclose(prior.logprob_many(np.array([[1.23, 4.56, 4.55]])), [lp])
    assert_allclose(prior.grad(np.array([1.23, 4.56, 4.56])), 0)
    grad = prior.grad(np.array([1.23, 4.56, 4.55]))
    assert grad[1] > 0, grad
    assert grad[2] < 0, grad
    n = participants.sum()
    ones = np.ones((n, n)) / n
    H = (np.eye(n) - ones) / 0.01**2
    assert_allclose(prior.hessian(None), H)
