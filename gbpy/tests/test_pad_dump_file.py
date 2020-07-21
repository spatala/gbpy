
import numpy as np
import pytest
import gbmc_v0.pad_dump_file as pdf
import gbmc_v0.util_funcs as uf


@pytest.mark.parametrize('filename0, rCut, lat_par, non_p',
                         [("data/dump_1", 8.1, 4.05, 2),
                          ("data/dump_2", 8.1, 4.05, 1)])
def test_pad_dump_file(filename0, rCut, lat_par, non_p):
    data = uf.compute_ovito_data(filename0)
    arr = pdf.p_arr(non_p)
    GbRegion, GbIndex, GbWidth, w_bottom_SC, w_top_SC = pdf.GB_finder(data, lat_par, non_p)
    sim_cell = data.cell[...]
    sim_1vec = np.array(sim_cell[:, arr[0]])
    sim_2vec = np.array(sim_cell[:, arr[1]])

    p1_vec = np.array([sim_1vec[arr[0]], sim_1vec[arr[1]]])
    p2_vec = np.array([sim_2vec[arr[0]], sim_2vec[arr[1]]])
    [n1, n2] = pdf.num_rep_2d(p1_vec, p2_vec, rCut)
    pts1, gb1_inds = pdf.pad_gb_perp(data, GbRegion, GbIndex, rCut, non_p)
    pts_w_imgs, gb1_inds, inds_arr = pdf.pad_dump_file(data, lat_par, rCut, non_p)

    area = np.linalg.norm(np.cross(sim_1vec, sim_2vec))
    num_atom_slice = np.power(np.sqrt(area) + 2 * rCut, 2) * np.shape(pts1)[0] / area
    err = 100 * (num_atom_slice - np.shape(pts_w_imgs)[0]) / np.shape(pts_w_imgs)[0]

    assert err < 5
