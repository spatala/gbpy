import numpy as np
import pytest
# import gbmc_v0.pad_dump_file as pad
import gbpy.pad_dump_file as pad
import gbpy.util_funcs as uf


@pytest.mark.parametrize('filename0, lat_par, num_GBregion, actual_min_z_gbreg, actual_max_z_gbreg,'
                         'actual_w_bottom_SC, actual_w_top_SC',
                         [("data/dump_1", 4.05, 138, -2.811127714, 2.811127714, 94, 91.5),
                          ("data/dump_2", 4.05, 51, -3.06795, 1.44512, 116.85, 118.462)])

def test_GB_finder(filename0, lat_par, num_GBregion, actual_min_z_gbreg, actual_max_z_gbreg,
                   actual_w_bottom_SC, actual_w_top_SC):
    data = uf.compute_ovito_data(filename0)
    non_p = uf.identify_pbc(data)
    GbRegion, GbIndex, GbWidth, w_bottom_SC, w_top_SC = pad.GB_finder(data, lat_par, non_p)

    assert np.abs((actual_w_bottom_SC - w_bottom_SC)/actual_w_bottom_SC) < .5
    assert np.abs((actual_w_top_SC - w_top_SC)/actual_w_top_SC) < .5
    assert np.abs(GbRegion[0] - actual_min_z_gbreg) < 1e-3
    assert np.abs(GbRegion[1] - actual_max_z_gbreg) < 1e-3
    assert np.shape(GbIndex)[0] == num_GBregion
