#!/usr/bin/env python
from numpy cimport int

#library functions that interact with lapack clibs

cdef extern void dgesdd_(char *jobz, int *m, int *n,
                        double a[], int *lda, double s[], double u[],
                        int *ldu, double vt[], int *ldvt, double work[],
                        int *lwork, int iwork[], int *info)

cdef extern void dgesvd_(char *jobu, char *jobvt,  int *m, int *n,
                        double a[], int *lda, double s[], double u[],
                        int *ldu, double vt[], int *ldvt, double work[],
                        int *lwork, int *info)

