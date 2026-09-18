"""pipeline.quiet() must silence noise WITHOUT silencing the row-alignment complaint.

quiet() exists because PHATE/lightning/scanpy are loud. But it did
`warnings.simplefilter("ignore")`, and the only row-alignment check anywhere downstream
is a UserWarning in manylatents' PrecomputedDataModule — so the one signal that a matrix
and its metadata disagree was being thrown away by the very wrapper meant to hide chatter.
"""
from __future__ import annotations

import warnings

import pytest

from manyruns.pipeline import quiet

# verbatim from manylatents/data/precomputed_datamodule.py:99
ALIGNMENT_WARNING = (
    "Metadata length (99) does not match embeddings length (100). "
    "Ensure metadata rows align with embedding rows."
)


def test_quiet_still_silences_ordinary_noise():
    with quiet():
        warnings.warn("phate: deprecated kwarg", UserWarning)
        warnings.warn("scanpy: future change", FutureWarning)
        warnings.warn("numpy: invalid value", RuntimeWarning)
    # no raise, nothing propagates — quiet() still does its job


def test_quiet_promotes_the_alignment_warning_to_an_exception():
    """The guard that was being swallowed. Misalignment is a bug, not noise."""
    with pytest.raises(UserWarning, match="does not match embeddings length"):
        with quiet():
            warnings.warn(ALIGNMENT_WARNING, UserWarning)


def test_the_promotion_does_not_leak_out_of_the_context():
    with pytest.raises(UserWarning):
        with quiet():
            warnings.warn(ALIGNMENT_WARNING, UserWarning)
    # quiet() uses catch_warnings, so the filter is restored on exit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.warn(ALIGNMENT_WARNING, UserWarning)  # must NOT raise out here
