import numpy as np
import pickle as pkl
# import os
# from ..lammps_input import write_lammps_dump
import ovito.io as oio
# dirname = os.path.dirname(__file__)
# filename = os.path.join(dirname, 'relative/path/to/file/you/want')


def test_create_pkldump(dump_name, path):
    """
    Function create the pickle file which containd the simulation cell, the coordinates of upper and lower grains.

    Parameters
    ------------
    dump_name :
        The name of the lammps dump file.
    path :
        The path that the pickle file will be saved.

    Returns
    ----------
    creates the "gb_attr.pkl".
    """
    pipeline = oio.import_file(str(dump_name), sort_particles=True)
    data = pipeline.compute()
    ID = data.particles['Particle Type'][...]
    pos = data.particles['Position'][...]
    ID_upper = np.where(ID == 1)[0]
    ID_lower = np.where(ID == 2)[0]
    u_pts = pos[ID_upper]
    l_pts = pos[ID_lower]
    sim_cell = data.cell[...]
    gb_attr = {}
    gb_attr['cell'] = sim_cell
    gb_attr['upts'] = u_pts
    gb_attr['lpts'] = l_pts
    output = open(path + 'gb_attr.pkl', 'wb')
    pkl.dump(gb_attr, output)
    output.close()


# test_create_pkldump('./data/dump_4', './data/')
box_bound, dump_lamp, box_type = lammps_box('./tests/data/gb_attr.pkl')
# write_lammps_dump("./tests/data/dump_1", box_bound, dump_lamp)
# write_lammps_script('./data/dump_1', '../lammps_dump/', 'in.minimize0', box_bound)
# lammps_exe_path = '/home/leila/Downloads/lammps-stable/lammps-7Aug19/src/lmp_mpi'
# os.system(str(lammps_exe_path) + '< ../lammps_dump/' + 'in.minimize0')

# box_bound, dump_lamp = lammps_box('./tests/data/gb_attr.pkl')
# write_lammps("./tests/data/dump_1", box_bound, dump_lamp)
