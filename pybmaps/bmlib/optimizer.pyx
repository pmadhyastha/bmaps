#!/usr/bin/env python
# cython: profile=True


from __future__ import division
import numpy as np
cimport numpy as np
import cython
from lsvd import fastsvd

cdef class Fista:

    cdef double lipschitz, tau, nu, lam_k
    cdef np.ndarray x_k

    def __cinit__(self, lipschitz, tau, x_k, lam_k=1):

        self.lipschitz = lipschitz
        self.tau = tau
        self.nu = (1 / self.lipschitz)
        self.x_k = x_k
        self.lam_k = lam_k

    cpdef fista_nn(self, np.ndarray ytemp):
        cdef np.ndarray[double, ndim=2] u
        cdef np.ndarray[double, ndim=1] s
        cdef np.ndarray[double, ndim=2] vt
        cdef int m = ytemp.shape[0]
        cdef int n = ytemp.shape[1]
        cdef np.ndarray[double, ndim=2] S = np.zeros((m,n))
        cdef int sing_val
        u, s, vt = fastsvd(ytemp)
        sing_val = s.shape[0]
        s = np.maximum(s - self.nu, 0)
        S[:sing_val, :sing_val] = np.diag(s)

        return (np.dot(u, np.dot(S, vt))), s

    cpdef np.ndarray fista_l1(self, np.ndarray ytemp):

        return np.multiply(np.sign(ytemp), np.max(np.abs(ytemp) -self.nu, 0))

    cpdef fista_l2(self, np.ndarray ytemp):

        return ytemp / (1 + self.nu)

    cpdef optimize(self, np.ndarray y_k, np.ndarray gYin, str reg_type):

        cdef np.ndarray[double, ndim=2] gY = self.tau * gYin
        cdef np.ndarray[double, ndim=2] ytemp = y_k - (gY/self.lipschitz)

        if reg_type == 'nn':
            x_kp1, s = self.fista_nn(ytemp)
        elif reg_type == 'l1':
            x_kp1 = self.fista_l1(ytemp)
        elif reg_type == 'l2':
            x_kp1 = self.fista_l2(ytemp)

        lam_kp1 = (1 + np.sqrt(1 + 4 * (self.lam_k**2))) / 2

        lr = (self.lam_k - 1) / lam_kp1
        y_kp1 = x_kp1 + lr * (x_kp1 - self.x_k)

        self.x_k = x_kp1
        self.lam_k = lam_kp1
        
        if reg_type == 'nn':
            return y_kp1, gY, s
        else:
            return y_kp1, gY


cdef class Fobos:

    cdef double eta, tau, iteration, lr

    def __cinit__(self, eta, tau):

        self.eta = eta
        self.tau = tau
        self.iteration = 1
        self.lr = self.eta / np.sqrt(self.iteration)

    cpdef fobos_nn(self, np.ndarray w_k1):
        cdef np.ndarray[double, ndim=2] u
        cdef np.ndarray[double, ndim=1] s
        cdef np.ndarray[double, ndim=2] vt
        cdef int m = w_k1.shape[0]
        cdef int n = w_k1.shape[1]
        cdef np.ndarray[double, ndim=2] S = np.zeros((m,n))
        cdef int sing_val
        cdef double nu = self.tau * self.lr
        u, s, vt = fastsvd(w_k1)
        sing_val = s.shape[0]
        s = np.maximum(s - nu, 0)
        S[:sing_val, :sing_val] = np.diag(s)

        return (np.dot(u, np.dot(S, vt))), s

    cpdef np.ndarray fobos_l1(self, np.ndarray w_k1):
        cdef double nu = self.lr * self.tau
        return np.multiply(np.sign(w_k1), np.max(np.abs(w_k1) - nu, 0))

    cpdef fobos_l2(self, np.ndarray w_k1):
        cdef double nu = self.lr * self.tau
        return w_k1 / (1 + nu)

    cpdef optimize(self, np.ndarray w_k, np.ndarray gY, str reg_type):

        self.lr = self.eta / np.sqrt(self.iteration)

        cdef np.ndarray[double, ndim=2] w_k1 = w_k - self.lr * gY

        if reg_type == 'nn':
            w_k, s = self.fobos_nn(w_k1)
        elif reg_type == 'l1':
            w_k = self.fobos_l1(w_k1)
        elif reg_type == 'l2':
            w_k = self.fobos_l2(w_k1)

        self.iteration = self.iteration + 1
        
        if reg_type == 'nn':
            return w_k, gY, s
        else:
            return w_k, gY







