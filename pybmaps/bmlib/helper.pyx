#!/usr/bin/env python
# cython: profile=True

from __future__ import division
import sys
import numpy as np
cimport numpy as np
from lsvd import fastsvd

cdef extern from "logsumexp.h":
    float flogsumexp(const float* buf, int N) nogil

cpdef double logsumexp(np.ndarray[dtype=double] ar):
    cdef np.ndarray[dtype=np.float32_t] a = np.float32(ar)
    """
    Compute the log of the sum of exponentials of input elements.

    Parameters
    ----------
    a : np.ndarray
        Input data. Must be contiguous.
    """
    if not (a.flags['C_CONTIGUOUS'] or a.flags['F_CONTIGUOUS']):
        raise TypeError('a must be contiguous')
    
    return np.double(flogsumexp(&a[0], a.size))

cpdef double fnorm(np.ndarray mat):
    ##frobenius norm
    axes = tuple(range(mat.ndim))
    return np.sqrt(np.add.reduce((mat.conj() * mat).real, axis=axes))

cpdef double snorm(np.ndarray mat):
    ##singular value norm - pass singular values only 
    return np.sum(sum(mat))

cpdef double l1norm(np.ndarray mat):
    ##l1 norm
    return np.add.reduce(np.abs(mat)).max()

cpdef double l2norm(np.ndarray mat):
    try: 
        mat.ndim > 1
        return fastsvd(mat)[1].max()
    except:
        return np.sqrt(np.add.reduce((mat.conj() * mat).real))

cpdef np.ndarray[double, ndim=2] normalize(np.ndarray[double, ndim=2] mat):
    ##row-wize normalization using f-norm
    cdef int rows = mat.shape[0]
    cdef int cols = mat.shape[1]
    cdef double den 
    cdef np.ndarray[double, ndim=2] res = np.zeros((rows, cols))

    for i in xrange(rows):
        den = fnorm(mat[i, :])
        try:
            assert den > 0 
            res[i] = mat[i, :] / den
        except:
            continue

    return res 




