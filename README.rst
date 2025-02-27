=========================
Optimized Nested Sampling
=========================

Faster inference by parameter space reduction of linear parameters.

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
