# %%
# https://web.stanford.edu/~wfsharpe/mia/opt/mia_opt3.htm
# L = q * w.b - w.A.w/2 subject to constraints sum(w)=1 and lb<=w<=ub.
import numpy as np

def make_array(x, shape):
    if np.isscalar(x):
        return np.full(shape, x)
    else:
        return np.asarray(x)

def find_max_return_opt(b, lb=0.0, ub=1.0):
    n = b.shape[-1]
    lb = make_array(lb, n)
    ub = make_array(ub, n)
    w = lb.copy()
    left_over = 1 - w.sum()
    arg_sorted = np.argsort(b)[::-1]
    loop = 0
    while left_over > 0:
        i = arg_sorted[loop]
        to_add = min(left_over, ub[i]-w[i])
        w[i] += to_add
        left_over -= to_add
        loop += 1
    return w

def find_status(w, lb, ub, nround=8):
    w = w.round(nround)
    return np.select([w<=lb, w>=ub, True], [-1, +1, 0])

def augment_matrices(A, b):
    """Convert
    A . w = q * b  - lambda * e
    to
    D . [w ; -lambda] = k + q * f
    """
    ones = np.ones_like(b)
    zeros = np.zeros_like(b)
    Du = np.hstack((A, ones[:, np.newaxis]))
    Dl = np.append(ones, 0)
    D = np.vstack((Du, Dl[np.newaxis]))
    k = np.append(zeros, 1)
    f = np.append(b, 0)
    return D, k, f

def replace_with_hard_bounds(D, k, f, status, lb=0, ub=1):
    n = k.shape[0] - 1
    zeros = np.zeros(n)
    eyez = np.hstack((np.eye(n), zeros[:, np.newaxis]))
    ub = make_array(ub, n)
    lb = make_array(lb, n)
    bounded_up = status==1
    bounded_down = status==-1
    bounded = bounded_up | bounded_down
    Dd = D.copy()
    kk = k.copy()
    ff = f.copy()
    Dd[:-1][bounded] = eyez[bounded]
    kk[:-1][bounded_up] = ub[bounded_up]
    kk[:-1][bounded_down] = lb[bounded_down]
    ff[:-1][bounded] = 0
    return Dd, kk, ff

def augment_bounded_matrices(w, A, b, lb=0, ub=1, status=None):
    """When bound is satisfied, replace the row equation with bounds.
    row equations:
    D . [w ; -lambda] = k + q * f
    bounds:
    Di -> [deltaij;0]
    ki -> lb/ub
    fi -> 0
    """
    D, k, f = augment_matrices(A, b)
    status = status if status is not None else find_status(w, lb, ub)
    Dd, kk, ff = replace_with_hard_bounds(D, k, f, status, lb=lb, ub=ub)
    return Dd, kk, ff

def get_critical_line(q, w, A, b, lb=0.0, ub=1.0, status=None):
    Dd, kk, ff = augment_bounded_matrices(w, A, b, lb=lb, ub=ub, status=status)
    DDinv = np.linalg.inv(Dd)
    xa = DDinv.dot(kk)
    xb = DDinv.dot(ff)
    return xa, xb

def get_lagrangian_derivative(q, w, A, b, lb=0.0, ub=1.0, status=None):
    D, k, f = augment_matrices(A, b)
    xa, xb = get_critical_line(q, w, A, b, lb=lb, ub=ub, status=status)
    dLa = - D.dot(xa)
    dLb = f - D.dot(xb)
    return dLa, dLb
    
def lower_corner(q, w, A, b, lb=0.0, ub=1.0, status=None, nround=8):
    lb, ub = make_array(lb, w.shape[0]), make_array(ub, w.shape[0])
    status = status if status is not None else find_status(w, lb, ub)
    xa, xb = get_critical_line(q, w, A, b, lb=lb, ub=ub, status=status)
    print(f"x = {xa.round(nround)} + q * {xb.round(nround)}")
    dLa, dLb = get_lagrangian_derivative(q, w, A, b, lb=lb, ub=ub, status=status)
    print(f"dL = {dLa.round(nround)} + q * {dLb.round(nround)}")
    qs = []
    for i, s in enumerate(status):
        if s==0:
            # in variable might become up or down
            if xb[i] < 0:
                # in variable may become up
                # ub[i] = xa[i] + q * xb[i]
                qi = (ub[i] - xa[i])/xb[i]
            elif xb[i] > 0:
                # in variable may become down
                qi = (lb[i] - xa[i])/xb[i]
            else:
                continue
        else:
            # up/down variable may become in
            qi = -dLa[i]/dLb[i]
        if np.round(qi, nround)<np.round(q, nround):
            qs.append((i, qi))
    i, qi = max(qs, key=lambda x:x[1], default=(None, 0))
    xi = xa + qi * xb
    wi = xi[:-1]
    print(f"Received {q} {w} {status} and got candidates {qs} selecting {i} {qi} {wi}")
    return i, qi, wi
    
def calc_corners(A, b, lb=0.0, ub=1.0):
    qs = []
    ws = []
    q = np.inf
    lb, ub = make_array(lb, b.shape[0]), make_array(ub, b.shape[0])
    w = find_max_return_opt(b, lb=lb, ub=ub)
    status0 = find_status(w, lb, ub)
    i = 0
    while (q is not None) and q>=0 and i is not None:
        qs.append(q)
        ws.append(w)
        i, q, w = lower_corner(q, w, A, b, lb=lb, ub=ub, status=status0)
        if i is not None:
            s = status0[i]
            if s==0:
                new_status = find_status(w, lb, ub)
                status0[i] = new_status[i]
            else:
                status0[i] = 0
            print(f"Appending {q} {w} and new status {status0}")
    return np.array(qs[::-1]), np.array(ws[::-1])
# %%
def example_vars():
    b = 1 + np.array([2.8, 6.3, 10.8])/100
    sd = np.array([1, 7.4, 15.4])/100
    rho = np.array([[1, 0.4, 0.15],[0.4, 1, 0.35], [0.15, 0.35, 1]])
    lb, ub = 0.2, 0.5
    C = sd[:, np.newaxis] * rho * sd[np.newaxis, :]
    return C, b, lb, ub
A, b, lb, ub = example_vars()
winf = find_max_return_opt(b, lb, ub)
assert np.allclose(winf, [0.2, 0.3, 0.5])
# compute_neighbours(np.inf, winf, A, b, lb, ub)
qs, ws = calc_corners(A, b, lb, ub)
