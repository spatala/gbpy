import pytest
import numpy as np
import gbmc_v0.util_funcs as uf
from ovito.data import NearestNeighborFinder


@pytest.mark.skip(reason="we can't push the data to repo. It is large.")
@pytest.mark.parametrize('filename0, lat_par, cut_off, num_neighbors, non_p_dir',
                         [('data/dump_1', 4.05, 10, 12, 2),
                          ('../lammps_dump/dump_after.1', 4.05, 10, 12, 2),
                          ('../lammps_dump/dump_befor.1', 4.05, 10, 12, 2)])
def test_check_a(filename0, lat_par, cut_off, num_neighbors, non_p_dir):
    # filename0 = 'data/dump_1'
    data = uf.compute_ovito_data(filename0)
    num = lat_par / np.sqrt(2)
    ptm_struct = data.particles['Structure Type'][...]
    position = data.particles['Position'][...]
    need_atom = np.where((position[:, non_p_dir] > 0) & (ptm_struct == 1))[0]
    pos_sc = position[need_atom, :]
    min_Z, max_Z = np.min(pos_sc[:, non_p_dir]) + cut_off, np.max(pos_sc[:, non_p_dir]) - cut_off

    area = np.where((position[:, non_p_dir] < max_Z) & (position[:, non_p_dir] > min_Z))[0]
    num_particl = np.shape(area)[0]
    finder = NearestNeighborFinder(num_neighbors, data)

    distances = np.zeros(num_particl)
    i = 0
    for index in area:
        # Iterate over the neighbors of the current particle, starting with the closest:
        for neigh in finder.find(index):
            distances[i] = neigh.distance + distances[i]
        i += 1

    cal_lat_par = np.mean(distances) / num_neighbors
    if np.abs(num - cal_lat_par) < 5 * 1e-3:
        err = 0

    assert err == 0
