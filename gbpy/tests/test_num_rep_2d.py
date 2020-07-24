import pytest
import numpy as np
import gbpy.pad_dump_file as pdf


@pytest.mark.parametrize('rCut, a_x, a_y, b_x, b_y, actual_nx, actual_ny',
                         [(4.2, 1, 0, 1, 1, 6, 5),
                          (1, 1, 0, 1, 1, 2, 1)
                          ])
def test_num_rep_2d(rCut, a_x, a_y, b_x, b_y, actual_nx, actual_ny):
    xvec = np.array([a_x, a_y])
    yvec = np.array([b_x, b_y])
    [nx, ny] = pdf.num_rep_2d(xvec, yvec, rCut)
    assert np.allclose(nx, actual_nx)
    assert np.allclose(ny, actual_ny)
