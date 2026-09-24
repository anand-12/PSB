"""Posterior-bridge transport for continual learning, in JAX for TPU sweeps.

Many small-MLP runs are packed into one program: a run axis (seeds and
hyperparameter settings, vmapped and sharded across devices) on top of a
particle axis (the K networks of one run).
"""
