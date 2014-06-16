#!/usr/bin/env python
# cython: profile = True

from __future__ import division
cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.int
ctypedef np.int_t DTYPE_t
@cython.boundscheck(False)
@cython.wraparound(False)
def evalEdge(np.ndarray score, np.ndarray truth):
    cdef int nQ = truth.shape[0]
    cdef int nC = truth.shape[1]
    cdef int pts = 0
    cdef int correct = 0
    cdef np.ndarray[DTYPE_t, ndim=1] rarr = np.array(range(nC), dtype=DTYPE)
    for q in xrange(nQ):
        nns = rarr[(truth[q] > 0)].tolist()
        not_nns = rarr[(truth[q] <= 0)].tolist()
        srow = score[q].tolist()
        for nn in nns:
            for not_nn in not_nns:
                pts += 1
                if srow[nn] > srow[not_nn]:
                    correct += 1 

    return (correct / pts) * 100


