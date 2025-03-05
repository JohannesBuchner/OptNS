=========================
Optimized Nested Sampling
=========================

Faster inference by parameter space reduction of linear parameters.


.. image:: https://img.shields.io/pypi/v/optns.svg
        :target: https://pypi.python.org/pypi/optns

.. image:: https://github.com/JohannesBuchner/OptNS/actions/workflows/tests.yml/badge.svg
        :target: https://github.com/JohannesBuchner/OptNS/actions/workflows/tests.yml

.. image:: https://coveralls.io/repos/github/JohannesBuchner/OptNS/badge.svg?branch=main
	:target: https://coveralls.io/github/JohannesBuchner/OptNS?branch=main

.. image:: https://img.shields.io/badge/GitHub-JohannesBuchner%2FOptNS-blue.svg?style=flat
        :target: https://github.com/JohannesBuchner/OptNS/
        :alt: Github repository

Context
-------

For models that are composed of additive components::

    y = A_1 * y_1(x|theta) + A_2 * y_2(x|theta) + ...

And data that are one of::

    y_obs ~ Normal(y, sigma)
    y_obs ~ Poisson(y)

y may be one or multi-dimensional.
sigma may be different for each y (heteroscadastic).

Here we see that each component y_i changes y linearly with its
normalisation parameter A_i.

We therefore have two groups of parameters:

 * linear parameters: A_i
 * non-linear parameters: theta

We can define the predictive part of our model as::

    y_1, y_2, ... = compute_components(x, theta)


What optns does
---------------

1. Profile likelihood inference with nested sampling. 
   That means the normalisations are optimized away.

2. Post-processing: The full posterior (A_i and theta) is sampled by 
   conditionally sampling A_i given theta.

Usage
-----

See the demo scripts in the examples folder!
