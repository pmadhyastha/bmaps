from __future__ import division
import numpy as np
import scipy.io as sio
import bmlib.learner as learner
import sys


qfile = sys.argv[1]
cfile = sys.argv[2]
trtfile = sys.argv[3]
vqfile = sys.argv[4]
vtfile = sys.argv[5]
lc = np.double(float(sys.argv[6]))
tau = np.double(float(sys.argv[7]))
maxiter = int(sys.argv[8])
reg = str(sys.argv[9])
regtype = str(sys.argv[10])
st = int(sys.argv[11])

query = np.array((sio.mmread(qfile)).todense())
cands = np.array((sio.mmread(cfile)).todense())
trtruth = np.array((sio.mmread(trtfile)).todense())
valquery = np.array((sio.mmread(vqfile)).todense())
valtruth = np.array((sio.mmread(vtfile)).todense())


#import pstats, cProfile

lrn = learner.Learn(query, cands, trtruth, valquery, valtruth, Wstart=None, valcand=None)
#cProfile.runctx("lrn.train_max_lik(lc, tau, maxiter, reg, regtype, st)", globals(), locals(), "Profile.prof")
weight,it = lrn.train_max_lik(lc, tau, maxiter, reg, regtype, st)
print 'saving last weight into matrix market format  . . . . '
fl = 'fin-'+reg+regtype+'-lc-'+str(lc)+'-tau-'+str(tau)+'-st-'+str(st)+'-iter-'+str(it)+'.mtx'
sio.mmwrite(fl, weight)
print 'Saved weight matrix'
print 'Exiting.....'
