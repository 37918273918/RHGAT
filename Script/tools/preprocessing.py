import numpy as np
import scipy.sparse as sp
import tensorflow as tf

def get_sparse_mat(a2b, a2idx, b2idx):
    n = len(a2idx)
    m = len(b2idx)
    assoc = np.zeros((n, m))
    for a, b_assoc in a2b.iteritems():
        if a not in a2idx:
            continue
        for b in b_assoc:
            if b not in b2idx:
                continue
            assoc[a2idx[a], b2idx[b]] = 1.
    assoc = sp.coo_matrix(assoc)
    return assoc

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    # init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    init_range = np.sqrt(3.0/(shape[0]+shape[1]))
    initial = tf.compat.v1.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.compat.v1.float32)
    # initial = tf.compat.v1.random_normal(shape, mean=0.0, stddev=np.sqrt(2.0/(shape[0]+shape[1])), dtype=tf.compat.v1.float32)
    return tf.compat.v1.Variable(initial, name=name)

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.compat.v1.zeros(shape, dtype=tf.compat.v1.float32)
    return tf.compat.v1.Variable(initial, name=name)