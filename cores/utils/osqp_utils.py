import osqp
import scipy
import numpy as np

def init_osqp(n_v, n_in, config, P_diag=None):
    if P_diag is None:
        P = scipy.sparse.eye(n_v, format='csc', dtype=config.np_dtype) 
    else:
        assert len(P_diag) == n_v
        P = scipy.sparse.csc_matrix(np.diag(P_diag))

    A_tmp = np.zeros((n_in, n_v), dtype=config.np_dtype)
    data = A_tmp.flatten()
    rows, cols = np.indices(A_tmp.shape)
    row_indices = rows.flatten()
    col_indices = cols.flatten()
    A = scipy.sparse.csc_matrix((data, (row_indices, col_indices)), shape=A_tmp.shape)

    l = np.full(n_in, -np.inf)
    u = np.full(n_in, np.inf)
    q = np.zeros(n_v, dtype=config.np_dtype)

    qp = osqp.OSQP()
    qp.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
    _ = qp.solve()

    return qp