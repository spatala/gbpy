import numpy as np
import pytest
import gbpy.pad_dump_file as pad
import gbpy.util_funcs as uf


@pytest.mark.parametrize('filename0, lat_par, non_p',
                         [("gbpy/tests/data/dump_1", 4.05, 2),
                          ("gbpy/tests/data/dump_2", 4.05, 1)])
def test_pad_gb_perp(filename0, lat_par, non_p):
    data = uf.compute_ovito_data(filename0)
    rCut = 2 * lat_par
    GbRegion, GbIndex, GbWidth, w_bottom_SC, w_top_SC = pad.GB_finder(data, lat_par, non_p, 'ptm', .1)
    pts1, gb1_inds = pad.pad_gb_perp(data, GbRegion, GbIndex, rCut, non_p)
    p_pos = pts1[gb1_inds]
    d_pos = data.particles['Position'][...][GbIndex, :]
    err = np.linalg.norm(p_pos - d_pos)
    assert np.allclose(0, err)
