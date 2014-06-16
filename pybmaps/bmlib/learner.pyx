#!/usr/bin/env python
# cython: profile=True

from __future__ import division
import numpy as np
cimport numpy as np
from time import time 
import cython
import optimizer 
from evalEdge import evalEdge
from helper import logsumexp, normalize, fnorm, l1norm, snorm

cdef class Learn:

    cdef np.ndarray query, cands, trtruth, valquery, valtruth, Wstart, valcand
    cdef int qrows, qcols, crows, ccols

    def __cinit__(self, query, cands, trtruth, valquery, valtruth, Wstart=None, valcand=None):

        self.query = normalize(query)
        self.cands = normalize(cands)
        self.valquery = normalize(valquery)
        self.trtruth = np.array(trtruth) 
        self.valtruth = np.array(valtruth)

        if valcand == None: 
            self.valcand = self.cands
        else:
            self.valcand = normalize(valcand)
    
        self.qrows = self.query.shape[0]
        self.qcols = self.query.shape[1]
        self.crows = self.cands.shape[0]
        self.ccols = self.cands.shape[1]

        if Wstart == None: 
            self.Wstart =  np.zeros((self.qcols, self.ccols))
        else:
            self.Wstart = Wstart 

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    cpdef train_max_lik(self, double lc, double tau, int maxiter, str reg, str regtype, int st):
        
        cdef np.ndarray[double, ndim=2] qd = self.query
        cdef np.ndarray[double, ndim=2] cd = self.cands
        cdef np.ndarray[double, ndim=2] vqd = self.valquery
        cdef np.ndarray[double, ndim=2] vcd = self.valcand
        cdef np.ndarray[double, ndim=2] ttr = self.trtruth
        cdef np.ndarray[double, ndim=2] vtr = self.valtruth
        cdef np.ndarray[double, ndim=2] x_k = self.Wstart
        cdef np.ndarray[double, ndim=2] y_k = self.Wstart
        cdef double eta = lc 
        if reg == 'fobos':
            optima = optimizer.Fobos(eta, tau)
        elif reg == 'fista':
            optima = optimizer.Fista(lc, tau, x_k)
        cdef int nqr = self.qrows
        cdef int nqc = self.qcols
        cdef int ncr = self.crows
        cdef int ncc = self.ccols
        cdef np.ndarray[double, ndim=2] score 
        cdef np.ndarray[double, ndim=2] score_val 
        cdef float res_train
        cdef float res_val
        cdef np.ndarray[double, ndim=2] gYin = np.zeros((nqc, ncc))
        cdef np.ndarray[double, ndim=2] gY
        cdef np.ndarray[double, ndim=2] logLin = np.zeros((nqr, ncr))
        cdef np.ndarray[double, ndim=2] logL
        cdef double nlik 
        cdef double zq
        cdef np.ndarray[double, ndim=1] pq 
        cdef np.ndarray[double, ndim=2] bigp
        cdef np.ndarray[double, ndim=1] av
        cdef np.ndarray[double, ndim=2] Exp_q
        cdef np.ndarray[double, ndim=2] Min = np.zeros((nqc, ncc))
        cdef np.ndarray[double, ndim=2] M
        cdef np.ndarray coin = np.array(range(nqr))
        cdef np.ndarray[double, ndim=2] LQ
        cdef np.ndarray[double, ndim=1] score_q
        cdef np.ndarray[double, ndim=2] qd_q
        cdef int sq 
        cdef np.ndarray[double, ndim=2] vq
        cdef double norm = 0
        cdef double normfin 
        cdef str fname
        cdef double start_loop 
        cdef double end_loop
        cdef float start = time()
        cdef float end 
        try:
            if st > 0: 
                for k in xrange(maxiter):
                    start_loop = time()
                    nlik = -1
                    score = np.dot(qd, np.dot(y_k, cd.transpose())) ##Inner Product
                    score_val = np.dot(vqd, np.dot(y_k, vcd.transpose()))

                    logL = logLin
                    for q in xrange(nqr):
                        zq = logsumexp(score[q, :].transpose())
                        for c in xrange(ncr):
                            logL[q,c] = ttr[q,c] * (score[q,c] -zq)

                    nlik = -1 * np.sum(sum(logL))


                    res_train = evalEdge(score, ttr)
                    res_val = evalEdge(score_val, vtr)

                    gY = gYin
                    
                    np.random.shuffle(coin)
                    LQ = qd[coin[:st], :]
                    
                    for s in xrange(st):
                        q = coin[s]
                        score_q = score[q, :]
                        zq = logsumexp(score_q)
                        pq = np.exp(score_q - zq)
                        bigp = (np.tile(pq, [ncc, 1])).transpose()
                        av = sum(np.multiply(cd, bigp))
                        vq = np.reshape(LQ[s, :], (-1,1))
                        Exp_q = vq * av
                        for c in xrange(ncr):
                            if ttr[q,c] > 0:
                                M = vq * cd[c, :]
                                gY = gY - ttr[q,c] * (M - Exp_q)
                    
                    if  regtype == 'nn':
                        y_k, gY, s = optima.optimize(y_k, gY, regtype)
                        normfin = snorm(s)
                    else:
                        y_k, gY = optima.optimize(y_k, gY, regtype)
                        if  regtype == 'l2':
                            normfin = fnorm(y_k)
                        else: 
                            normfin = l1norm(y_k)
                    end_loop = time()

                    fle = open('output.txt', 'a')
                    fle.write(' iter: %3d obj: %12.5f nlik: %12.5f %s norm: %9.5f avg_ranking_train: %6.3f, avg_ranking_val: %6.3f time: %4.3f\n' % \
                            (k+1, nlik+tau*norm, nlik, regtype, norm, res_train, res_val, end_loop-start_loop))
                    fle.close()
                    norm = normfin

                    if np.remainder(k+1, 10) == 0:
                        fname = 'modelFile-tau'+str(tau)+'-lc-o-eta'+str(lc)+'-st'+str(st)+'-iter'+str(k+1)+'.npy'
                        np.save(fname, y_k)
                    else:
                       continue
            else:
                for k in xrange(maxiter):
                    start_loop = time()
                    nlik = -1

                    score = np.dot(qd, np.dot(y_k, cd.transpose())) ##Inner Product

                    logL = logLin
                    for q in xrange(nqr):
                        zq = logsumexp(score[q, :].transpose())
                        for c in xrange(ncr):
                            logL[q,c] = ttr[q,c] * (score[q,c] -zq)

                    nlik = -1 * np.sum(sum(logL))

                    score_val = np.dot(vqd, np.dot(y_k, vcd.transpose()))

                    res_train = evalEdge(score, ttr)
                    res_val = evalEdge(score_val, vtr)

                    gY = gYin
                    
                    for q in xrange(nqr):
                        score_q = score[q, :]
                        qd_q = np.reshape(qd[q, :], (-1,1))
                        zq = logsumexp(score_q)
                        pq = np.exp(score_q - zq)
                        bigp = (np.tile(pq, [ncc, 1])).transpose()
                        av = sum(np.multiply(cd, bigp))
                        Exp_q = qd_q * av
                        M = Min
                        for c in xrange(ncr):
                            if ttr[q,c] > 0:
                                M = qd_q * cd[c, :]
                                gY = gY - ttr[q,c] * (M - Exp_q)

                    if  regtype == 'nn':
                        y_k, gY, s = optima.optimize(y_k, gY, regtype)
                        normfin = snorm(s)
                    else:
                        y_k, gY = optima.optimize(y_k, gY, regtype)
                        if  regtype == 'l2':
                            normfin = fnorm(y_k)
                        else: 
                            normfin = l1norm(y_k)
                    end_loop = time()

                    fle = open('output.txt', 'a')
                    fle.write(' iter: %3d obj: %12.5f nlik: %12.5f %s norm: %9.5f avg_ranking_train: %6.3f, avg_ranking_val: %6.3f time: %4.3f\n' % \
                            (k+1, nlik+tau*norm, nlik, regtype, norm, res_train, res_val, end_loop-start_loop))
                    fle.close()
                    norm = normfin

                    if np.remainder(k+1, 10) != 0:
                        continue
                    else:
                        fname = 'model-'+str(reg)+str(regtype)+'-tau-'+str(tau)+'-lc-o-eta-'+str(lc)+'-st-'+str(st)+'-iter-'+str(k+1)+'.npy'
                        np.save(fname, y_k)
        except KeyboardInterrupt:
            print '....... Training stopped due to a keyboard interrupt .......'

        end = time()
        
        print 'total time taken = ', end - start 

        return y_k, k+1


