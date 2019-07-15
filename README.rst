==============
About torforce
==============


.. image:: https://img.shields.io/pypi/v/torforce.svg
        :target: https://pypi.python.org/pypi/torforce

.. image:: https://img.shields.io/travis/leaprovenzano/torforce.svg
        :target: https://travis-ci.org/leaprovenzano/torforce

.. image:: https://readthedocs.org/projects/torforce/badge/?version=latest
        :target: https://torforce.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status



torforce is a library for reinforcement learning in pytorch.


this library is currently in active development towards 0.1.0. until then nothing is stable and the main branch will be develop.


* Free software: MIT license


Features
--------

* Custom `UnimodalBeta` and `ScaledBeta` distributions for use in policies with constrained output spaces.
* `torforce.env_wrappers` for wrapping gym envs in a consistent interface pipelines and state tracking. `TensorEnvWrapper` handles tensor numpy conversion headaches
* Losses : `ClippedSurrogateLoss`
* Distribution Policy Layers: Policy Layers for learn a output a distribution over an action space (given some features)
