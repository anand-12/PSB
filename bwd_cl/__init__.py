"""Bayesian weight diffusion for replay-free continual learning.

The posterior over solutions is carried between tasks as a Gaussian envelope
(mean and diagonal precision) plus a particle population whose non-Gaussian
structure is modelled by a score network. Neither part grows with the stream.
"""

__version__ = "0.1.0"
