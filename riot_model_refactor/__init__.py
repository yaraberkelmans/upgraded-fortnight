"""riot_model_refactor package.

Spatially-vectorized variant of ``riot_model``. Behaviour is intended to be
bit-for-bit identical on the same seed; only the per-tick neighbour counting and
aggregate reporting are vectorized with numpy. See ``compare_refactor.py``.
"""
