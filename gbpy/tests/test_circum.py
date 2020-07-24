
import numpy as np
import pytest
from gbpy.vv_props import Circum_O_R


def point_gen(v_low, v_high):
    points = np.random.uniform(low=v_low, high=v_high, size=(12, )).reshape(4, -1)
    # Check the points are non planar
    AB = points[0, :] - points[1, :]
    AC = points[0, :] - points[2, :]
    Pnormal = np.cross(AB, AC)
    cte = np.dot(Pnormal, points[3, :]) - np.dot(Pnormal, points[0, :])
    while cte == 0:
        points[3, :] = np.random.uniform(low=v_low, high=v_high, size=(3, ))
        cte = np.dot(Pnormal, points[3, :]) - np.dot(Pnormal, points[0, :])

    return points


@pytest.mark.parametrize('test_num, tol, v_low, v_high',
                         [(1000, 1, -30, 30)])
def test_circum(test_num, tol, v_low, v_high):
    test_list_1 = [0]*test_num

    for i in range(test_num):
        points = point_gen(v_low, v_high)
        circum_center, circum_rad = Circum_O_R(points, tol)
        val = points - circum_center
        rad_val = np.linalg.norm(val, axis=1)
        err = np.sum(np.abs(rad_val - circum_rad))
        if circum_rad != 0:
            if err > 1e-1:
                test_list_1[i] = 1
            else:
                test_list_1[i] = 0

    test_1 = np.sum(test_list_1)
    assert test_1 == 0
