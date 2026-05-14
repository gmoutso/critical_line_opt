# %%
# Critical Line Algorithm
# min L = wT.A.w/2 - q * wT.b subject to constraints sum(w)=1 and lb<=w<=ub.
# dL/dw = A.w - q * b - lambda * b
# KKT condition dL/dw<=0 at w=ub, dL/dw>=0 at w=lb, dL/dw=0 in-bound.
# https://web.stanford.edu/~wfsharpe/mia/opt/mia_opt3.htm

import numpy as np

def make_array(x, shape):
    if np.isscalar(x):
        return np.full(shape, x)
    else:
        return np.asarray(x)

def make_array_like(x, y):
    if np.isscalar(x):
        return np.full_like(y, fill_value=x)
    else:
        return np.asarray(x)

def find_status(w, lb=0.0, ub=1.0, nround=8):
    lb = make_array_like(lb, w)
    ub = make_array_like(ub, w)
    atol = 10**(-nround)
    return np.select([w<lb , np.isclose(w, lb, atol=atol),
                      w>ub , np.isclose(w, ub, atol=atol),
                      True],
                     [-1, -1, +1, +1, 0])


def find_max_return(b, lb=0.0, ub=1.0):
    lb = make_array_like(lb, b)
    ub = make_array_like(ub, b)
    w = lb.copy()
    left_over = 1 - w.sum()
    arg_sorted = np.argsort(-b)
    loop = 0
    while left_over > 0:
        i = arg_sorted[loop]
        to_add = min(left_over, ub[i]-w[i])
        w[i] += to_add
        left_over -= to_add
        loop += 1
    return w

def find_max_return_enhanced(b, lb=0.0, ub=1.0):
    lb = make_array_like(lb, b)
    ub = make_array_like(ub, b)
    w = lb.copy()
    status = make_array_like(-1, b)
    left_over = 1 - w.sum()
    arg_sorted = np.argsort(-b)
    loop = 0
    while left_over > 0:
        i = arg_sorted[loop]
        to_add = min(left_over, ub[i]-w[i])
        w[i] += to_add
        left_over -= to_add
        if left_over>0:
            status[i]=1
        else:
            status[i]=0
        loop += 1
    return w, status
    

def augment_matrices(A, b):
    """Convert
    A . w = q * b  + lambda * e  & sum(w)=1
    to
    D . [w ; -lambda] = k + q * f
    """
    ones = np.ones_like(b)
    zeros = np.zeros_like(b)
    D = np.block([[A, ones[:, None]],
                  [ones[None, :], 0]])
    k = np.append(zeros, 1)
    f = np.append(b, 0)
    return D, k, f

def replace_with_hard_bounds(D, k, f, status, lb=0.0, ub=1.0):
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

def augment_bounded_matrices(w, A, b, lb=0.0, ub=1.0, status=None):
    """When bounds are satisfied, replace those row equation with hard bounds.
    row equations:
    D . [w ; -lambda] = k + q * f
    bounds on i:
    Di -> [deltaij;0]
    ki -> lb/ub
    fi -> 0
    """
    status = status if status is not None else find_status(w, lb, ub)
    D, k, f = augment_matrices(A, b)
    Dd, kk, ff = replace_with_hard_bounds(D, k, f, status, lb=lb, ub=ub)
    return Dd, kk, ff

def get_critical_line(w, A, b, lb=0.0, ub=1.0, status=None):
    Dd, kk, ff = augment_bounded_matrices(w, A, b, lb=lb, ub=ub, status=status)
    DDinv = np.linalg.inv(Dd)
    xa = DDinv.dot(kk)
    xb = DDinv.dot(ff)
    return xa, xb

def get_lagrangian_derivative(w, A, b, lb=0.0, ub=1.0, status=None):
    status = status if status is not None else find_status(w, lb, ub)
    # D, k, f = augment_bounded_matrices(w, A, b, lb=lb, ub=ub, status=status)
    D, k, f = augment_matrices(A, b) # same..
    xa, xb = get_critical_line(w, A, b, lb=lb, ub=ub, status=status)
    dLa = D.dot(xa)
    dLb = D.dot(xb) - f
    return dLa, dLb

# from enum import Enum
# AssetStatus = Enum("AssetStatus", "down in up")

def lower_corner_criteria(status, xa, xb, dLa, dLb, lb=0.0, ub=1.0):
    lb = make_array_like(lb, xa)
    ub = make_array_like(ub, xa)
    qs = []
    for i, s in enumerate(status):
        if s==0 and xb[i]<0:
            # in variable may become up
            # ub[i] = xa[i] + q * xb[i]
            qi = (ub[i] - xa[i])/xb[i]
            new_status = 1
        elif s==0 and xb[i]>0:
            # in variable may become down
            # lb[i] = xa[i] + q * xb[i]
            qi = (lb[i] - xa[i])/xb[i]
            new_status = -1
        elif s==1 and dLb[i]<0:
            # up variable may become in
            qi = -dLa[i]/dLb[i]
            new_status = 0
        elif s==-1 and dLb[i]>0:
            # up/down variable may become in
            qi = -dLa[i]/dLb[i]
            new_status = 0
        else:
            continue
        qs.append((i, qi, new_status))
    return qs

    
def lower_corner(q, w, A, b, lb=0.0, ub=1.0, status=None, nround=8):
    lb = make_array_like(lb, w)
    ub = make_array_like(ub, w)
    status = status.copy() if status is not None else find_status(w, lb, ub)
    xa, xb = get_critical_line(w, A, b, lb=lb, ub=ub, status=status)
    dLa, dLb = get_lagrangian_derivative(w, A, b, lb=lb, ub=ub, status=status)
    qs = lower_corner_criteria(status, xa, xb, dLa, dLb, lb, ub)
    qs = list(filter(lambda x: np.round(x[1], nround)<np.round(q, nround), qs))
    if len(qs)==0:
        return None, None, None, None
    i, qi, new_status = max(qs, key=lambda x:x[1], default=(None, None, None))
    xi = xa + qi * xb
    wi = xi[:-1]
    status[i] = new_status
    return i, qi, wi, status
    
def calc_corners(A, b, lb=0.0, ub=1.0, include_inf=False, include_zero=False):
    if len(b.shape)==2 and len(A.shape)==1:
        raise ValueError("b should be 1d and A should be 2d")
    qs = []
    ws = []
    q = np.inf
    lb, ub = make_array_like(lb, b), make_array_like(ub, b)
    w, status0 = find_max_return_enhanced(b, lb=lb, ub=ub)
    i = 0
    while (q is not None) and q>=0 and i is not None:
        qs.append(q)
        ws.append(w)
        i, q, w, status0 = lower_corner(q, w, A, b, lb=lb, ub=ub, status=status0)
    qs, ws = np.array(qs[::-1]), np.array(ws[::-1])
    if not include_inf:
        qs = qs[:-1]
        ws = ws[:-1]
    if include_zero:
        qs = np.append(0, qs)
        ws = np.vstack([ws[0], ws])
    return qs, ws
