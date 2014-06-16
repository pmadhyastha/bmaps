#!/usr/bin/env python

import numpy as np
import scipy.sparse as ss
import scipy.io as sio
import glob
import os
from bmlib.evalEdge import evalEdge

def compute(directory, qdevel, cdevel, constr_dev):
    qdevel = np.array(sio.mmread(qdevel).todense())
    cdevel = np.array(sio.mmread(cdevel).todense()).transpose()
    constrdevel = np.array(sio.mmread(constr_dev).todense())
    os.chdir(directory)
    for file in glob.glob('*.npy'):
        flname = file.split('.npy')[0]
        rep = np.load(file)
        sio.mmwrite(str(flname)+'.mtx', ss.coo_matrix(rep))
        mat = np.dot(qdevel, np.dot(rep, cdevel))
        score = evalEdge(mat, constrdevel)
        fle = open('new_output.txt', 'a')
        fle.write('file: '+flname+'\t'+str(score)+'\n')
        fle.close()

    return 'Done'

