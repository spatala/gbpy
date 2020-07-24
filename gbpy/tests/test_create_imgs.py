
import numpy as np
import pytest
import gbpy.pad_dump_file as pdf
import gbpy.util_funcs as uf


@pytest.mark.parametrize('filename0, rCut, lat_par, non_p',
                         [("gbpy/tests/data/dump_1", 8.1, 4.05, 2),
                          ("gbpy/tests/data/dump_1", 30, 4.05, 2),
                          ("gbpy/tests/data/dump_2", 8.1, 4.05, 1)])
def test_create_imgs(filename0, rCut, lat_par, non_p):
    data = uf.compute_ovito_data(filename0)
    arr = pdf.p_arr(non_p)
    GbRegion, GbIndex, GbWidth, w_bottom_SC, w_top_SC = pdf.GB_finder(data, lat_par, non_p, 'ptm', .1)
    sim_cell = data.cell[...]
    sim_1vec = np.array(sim_cell[:, arr[0]])
    sim_2vec = np.array(sim_cell[:, arr[1]])

    p1_vec = np.array([sim_1vec[arr[0]], sim_1vec[arr[1]]])
    p2_vec = np.array([sim_2vec[arr[0]], sim_2vec[arr[1]]])
    [n1, n2] = pdf.num_rep_2d(p1_vec, p2_vec, rCut)
    pts1, gb1_inds = pdf.pad_gb_perp(data, GbRegion, GbIndex, rCut, non_p)
    pts_w_imgs, inds_array = pdf.create_imgs(pts1, n1, n2, sim_1vec, sim_2vec, non_p)

    num0 = pts_w_imgs.shape[0]/pts1.shape[0]
    num1 = np.power(n1+n2+1, 2)
    assert np.allclose(num0, num1)
