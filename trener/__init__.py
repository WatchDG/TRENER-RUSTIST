"""Post-training pipeline for the JOLT and JOLT+ methods.

A single trainable model plays two roles — teacher (with privileged information
``c``) and student (task statement only); both roles update the shared
parameters with one update per step.

Formalization: ``docs/01-jolt.md`` (weights), ``docs/02-algorithm.md`` (step),
``docs/04-implementation.md`` (requirements).
"""

__version__ = "0.9.0"

__all__ = ["__version__"]
