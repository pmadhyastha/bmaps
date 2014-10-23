#!/usr/bin/env python

import Cython.Distutils
from distutils.extension import Extension
import distutils.core
import os
import numpy as np

pwd = os.path.dirname(__file__)
virtual_env = os.environ.get('VIRTUAL_ENV', '')
includes = [os.path.join(pwd, 'bmlib'), os.path.join(pwd, 'bmlib/logsumexp'), os.path.join(virtual_env, 'include')]
exts = [
        Extension('bmlib.lsvd', ["bmlib/lsvd.pyx"], libraries=['lapack'], include_dirs=includes),
        Extension('bmlib.helper', ["bmlib/helper.pyx", "bmlib/logsumexp/logsumexp.c"],
            include_dirs=includes, extra_compile_args=['-msse2']),
        Extension('bmlib.evalEdge', ["bmlib/evalEdge.pyx"], include_dirs=includes),
        Extension('bmlib.optimizer', ["bmlib/optimizer.pyx"], include_dirs=includes),
        Extension('bmlib.learner', ["bmlib/learner.pyx"], include_dirs=includes)
        ]

distutils.core.setup(
        name='bmlib-0.1',
        cmdclass={'build_ext': Cython.Distutils.build_ext},
        packages=['bmlib'],
        ext_modules=exts,
        include_dirs=[np.get_include()]
)

