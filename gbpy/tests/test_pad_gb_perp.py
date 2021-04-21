import numpy as np
import pytest
import gbpy.pad_dump_file as pad
import gbpy.util_funcs as uf
import byxtal.lattice as gbl

@pytest.mark.parametrize('filename0, element, non_p',
                         [("tests/data/dump_1", "Al", 2),
                          ("tests/data/dump_2", "Al", 1)])
def test_pad_gb_perp(filename0, element, non_p):
    l1 = gbl.Lattice(str(element))
    data = uf.compute_ovito_data(filename0)
    rCut = 2 * l1.lat_params['a']
    GbRegion, GbIndex, GbWidth, w_bottom_SC, w_top_SC = pad.GB_finder(data, l1, non_p, 'ptm', .1)
    pts1, gb1_inds = pad.pad_gb_perp(data, GbRegion, GbIndex, rCut, non_p)
    p_pos = pts1[gb1_inds]
    d_pos = data.particles['Position'][...][GbIndex, :]
    err = np.linalg.norm(p_pos - d_pos)
    assert np.allclose(0, err)
