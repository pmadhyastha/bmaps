#!/usr/bin/env python
# cython: profile=True
cimport cython
import numpy as np
cimport numpy as np
from clapack cimport dgesdd_, dgesvd_

#The dgesdd subroutine is better as it is considerably faster
# Refer http://www.netlib.org/lapack/lug/node71.html for benchmarks

#dgesdd uses a divide and conquer algorithm for the bidiagonal SVD, whereas
#dgesvd uses a QR algorithm.

#http://www.netlib.org/lapack/lawnspdf/lawn88.pdf 
#One would choose the implementation based upon the QR algorithm (dgesvd) rather than
#the one which uses a Divide and Conquer algorithm (dgesdd) as dgesdd
#requires O(min(m,n)^2) memory, where k=min(m,n), while the dgesvd 
#requires O(max(m,n)).

#fastsvd uses dgesdd and stdsvd uses dgesvd

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int int_max(int a, int b): return a if a >= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int int_min(int a, int b): return a if a <= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
def fastsvd(mat):
    cdef np.ndarray[double, ndim=2] a = np.asfortranarray(mat)
    cdef int M = a.shape[0]
    cdef int N = a.shape[1]
    cdef int a_max = int_max(M,N)
    cdef int a_min = int_min(M,N)
    cdef np.ndarray[double, ndim=1] s = np.empty(a_min, dtype=np.double, order='F')
    cdef np.ndarray[double, ndim=2] u = np.empty((M, M), dtype=np.double, order='F')
    cdef np.ndarray[double, ndim=2] vt = np.empty((N, N), dtype=np.double, order='F')
    cdef int info = 0
    cdef char jobz = 'A'
    cdef int lwork = a_min*(6+4*a_min)+a_max
    cdef np.ndarray[double, ndim=1] work = np.zeros((lwork,), dtype=np.double, order='F')
    cdef int tiwork = 8*a_min
    cdef np.ndarray iwork = np.zeros((tiwork,), dtype=int, order='F')
    cdef int ldvt = N
    cdef int ldu = M
    cdef int lda = M 
    dgesdd_(&jobz, &M, &N, <double *> a.data, &lda, 
            <double *> s.data, <double *> u.data, &ldu, <double *> vt.data, 
            &ldvt, <double *> work.data, &lwork, <int *> iwork.data, &info)

    
    return u, s, vt 


@cython.boundscheck(False)
@cython.wraparound(False)
def stdsvd(mat):
    cdef np.ndarray[double, ndim=2] a = np.asfortranarray(mat)
    cdef int M = a.shape[0]
    cdef int N = a.shape[1]
    cdef int a_max = int_max(M,N)
    cdef int a_min = int_min(M,N)
    cdef np.ndarray[double, ndim=1] s = np.empty(a_min, dtype=np.double, order='F')
    cdef np.ndarray[double, ndim=2] u = np.empty((M, M), dtype=np.double, order='F')
    cdef np.ndarray[double, ndim=2] vt = np.empty((N, N), dtype=np.double, order='F')
    cdef int info = 0
    cdef char jobu = 'A'
    cdef char jobvt = 'A'
    cdef int lwork = int_max(3*a_min+a_max, 5*a_min)
    cdef np.ndarray[double, ndim=1] work = np.zeros((lwork,), dtype=np.double, order='F')
    cdef int ldvt = N
    cdef int ldu = M
    cdef int lda = M 
    dgesvd_(&jobu, &jobvt, &M, &N, <double *> a.data, &lda, 
            <double *> s.data, <double *> u.data, &ldu, <double *> vt.data, 
            &ldvt, <double *> work.data, &lwork, &info)

    return u, s, vt

