# %%
import pytest

# import fixtures
import numpy as np
import critical_line_opt
from critical_line_opt import (
    find_max_return,
    find_status,
    make_array_like,
    calc_corners,
    get_lagrangian_derivative,
    get_critical_line,
    lower_corner_criteria,
    lower_corner,
)


def example_vars1():
    b = np.array([2.8, 6.3, 10.8])
    sd = np.array([1, 7.4, 15.4])
    rho = np.array([[1, 0.4, 0.15], [0.4, 1, 0.35], [0.15, 0.35, 1]])
    lb, ub = 0.2, 0.5
    C = sd[:, np.newaxis] * rho * sd[np.newaxis, :]
    return 2 * C, b, lb, ub


def test_sharpe_example1():
    _, b, lb, ub = example_vars1()
    winf = find_max_return(b, lb, ub)
    assert np.allclose(winf, [0.2, 0.3, 0.5])


def test_sharpe_example2():
    C, b, lb, ub = example_vars1()
    w = np.array([0.2, 0.3, 0.5])
    dLa, dLb = get_lagrangian_derivative(w, C, b, lb, ub)
    dL = (dLa + 45 * dLb).round(4)
    assert np.allclose(dL, [88.06, 0, -14.4104, 1])


def test_sharpe_example3():
    C, b, lb, ub = example_vars1()
    w = np.array([0.2, 0.3, 0.5])
    xa, xb = get_critical_line(w, C, b, lb, ub)
    dLa, dLb = get_lagrangian_derivative(w, C, b, lb, ub)
    y = xa + 44 * xb
    dL = dLa + 44 * dLb
    yi = np.array([0.2, 0.3, 0.5, 203.2740])
    dLi = np.array([84.56, 0, -9.9104, 1])
    assert np.allclose(y, yi, atol=1e-4)
    assert np.allclose(dL, dLi, atol=1e-4)


def test_sharpe_example4():
    C, b, lb, ub = example_vars1()
    w = np.array([0.2, 0.3, 0.5])
    xa, xb = get_critical_line(w, C, b, lb, ub)
    dLa, dLb = get_lagrangian_derivative(w, C, b, lb, ub)
    assert np.allclose(xa, [0.2, 0.3, 0.5, -73.9260], atol=1e-4)
    assert np.allclose(xb, [0, 0, 0, 6.3], atol=1e-4)
    assert np.allclose(dLa, [-69.44, 0, 188.0896, 1], atol=1e-4)
    assert np.allclose(dLb, [3.5, 0, -4.5, 0], atol=1e-4)


def test_lower_corner_criteria():
    C, b, lb, ub = example_vars1()
    w = np.array([0.2, 0.3, 0.5])
    status = np.array([-1, 0, 1])
    xa = np.array([0.2, 0.3, 0.5, -73.9260 / 2])
    xb = np.array([0, 0, 0, 6.3])
    dLa = np.array([-69.44, 0, 188.0896, 2]) / 2
    dLb = np.array([3.5, 0, -4.5, 0])
    qs = lower_corner_criteria(status, xa, xb, dLa, dLb, lb=lb, ub=ub)
    qs = [x[1] for x in qs]
    assert np.allclose(qs, [19.84 / 2, 41.7977 / 2], atol=1e-4)


def test_calc_corners_criteria_called():
    C, b, lb, ub = example_vars1()
    w = np.array([0.2, 0.3, 0.5])
    lb = make_array_like(lb, w)
    ub = make_array_like(ub, w)
    status = np.array([-1, 0, 1])
    xa, xb = get_critical_line(w, C, b, lb=lb, ub=ub, status=status)
    dLa, dLb = get_lagrangian_derivative(w, C, b, lb=lb, ub=ub, status=status)
    qs = lower_corner_criteria(status, xa, xb, dLa, dLb, lb, ub)
    qs = [x[1] for x in qs]
    assert np.allclose(qs, [19.84, 41.7977], atol=1e-4)


def test_sharpe_example5():
    C, b, lb, ub = example_vars1()
    qs, ws = calc_corners(C, b, lb=lb, ub=ub)
    assert np.allclose(
        qs,
        [13.7344, 15.1038434, 21.02177625, 22.295, 22.94008889, 41.79768889, np.inf],
        atol=1e-4,
    )
    assert np.allclose(
        ws,
        [
            [0.5, 0.3, 0.2],
            [0.45191561, 0.34808439, 0.2],
            [0.22180738, 0.5, 0.27819262],
            [0.2, 0.5, 0.3],
            [0.2, 0.5, 0.3],
            [0.2, 0.3, 0.5],
            [0.2, 0.3, 0.5],
        ],
        atol=1e-4,
    )
