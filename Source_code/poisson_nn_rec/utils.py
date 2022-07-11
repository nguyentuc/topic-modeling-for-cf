import numpy as np
import math

def load_matrix_np(path):
    return np.loadtxt(path,dtype=float)

def norm_matrix(matrix):
    row_sums = matrix.sum(axis=1)
    matrix /= row_sums[:,np.newaxis]
    return matrix

def vnormalize(x):
    v = np.sum(x);
    if np.isnan(v):
        print ("WTF???")
        exit(-1);

    if math.fabs(v)>1e-40:
        x  *= 1.0/v
    return x

def numpy_add_row(math_source,vector,row_id):
        math_source[row_id] += vector


def numpy_copy_vector(v_source,v_des):
    sz = len(v_source)
    for i in xrange(sz):
        v_des[i] = v_source[i]

def swap(a,b):
    return b,a