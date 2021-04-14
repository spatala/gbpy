import util_funcs as uf
import pad_dump_file as pdf
import vv_props as vvp
import lammps_dump_writer as ldw
import lammps_script_writer as lsw
import ovito.data as ovd
from ovito.pipeline import StaticSource, Pipeline
import ovito.modifiers as ovm
from shutil import copyfile
import numpy as np

#  --------------------------------------------------------------------------
#  Define the input
#  --------------------------------------------------------------------------
lat_par = 4.05
rCut = 2*lat_par
CohEng= -3.35999998818377  #  calculated from in.cohesive
Tm = 933.5
weight_1 = .5
tol_fix_reg = 5 * lat_par  # the width of rigid traslation region
SC_tol = 5 * lat_par
# str_alg = "ptm"
str_alg = "csc"
csc_tol = .1
method = "anneal"
# method = "min"
Etol_val=1e-25
Ftol_val=1e-25
if method=="anneal":
    Etol_val0=1e-5
    Ftol_val0=1e-5
else:
    Etol_val0=1e-25
    Ftol_val0=1e-25

MaxIter_val=5000
MaxEval_val=10000
Iter_heat_val=1000
Iter_equil_val=3000
Iter_cool_val=1000

filename_0 = 'dump'
data_0 = uf.compute_ovito_data(filename_0)
non_p = uf.identify_pbc(data_0)
#  find the gb atoms
GbRegion, GbIndex, GbWidth, w_bottom_SC, w_top_SC = pdf.GB_finder(data_0, lat_par, non_p, str_alg, csc_tol)


pts_w_imgs, gb1_inds, inds_arr = pdf.pad_dump_file(data_0, lat_par, rCut, non_p, str_alg, csc_tol)
tri_vertices, gb_tri_inds = vvp.triang_inds(pts_w_imgs, gb1_inds, inds_arr)
cc_coors, cc_rad = vvp.vv_props(pts_w_imgs, tri_vertices, gb_tri_inds, lat_par)
cc_coors1 = vvp.wrap_cc(data_0.cell, cc_coors)
Prob = uf.radi_normaliz(cc_rad)