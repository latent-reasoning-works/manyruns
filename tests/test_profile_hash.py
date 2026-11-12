import hashlib
import numpy as np
from scipy import sparse
from manyruns.pipeline.features import input_sha256


def test_dense_hash_uses_canonical_original_matrix():
    x = np.arange(12, dtype=np.float32).reshape(4,3)
    expected = hashlib.sha256(b'manyruns-array/1\nfloat64\n4,3\n' + x.astype('float64').tobytes()).hexdigest()
    assert input_sha256(x) == expected
    assert input_sha256(x[:, ::-1][:, ::-1]) == expected


def test_sparse_hash_never_densifies(monkeypatch):
    x = sparse.csr_matrix(np.eye(4))
    monkeypatch.setattr(sparse.csr_matrix, 'toarray', lambda *a, **k: (_ for _ in ()).throw(AssertionError('densified')))
    assert len(input_sha256(x)) == 64
    assert input_sha256(x) == input_sha256(x.tocoo())
