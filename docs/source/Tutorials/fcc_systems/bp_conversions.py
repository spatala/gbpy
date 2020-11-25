import byxtal.find_csl_dsc as fcd
import numpy.linalg as nla
import byxtal.integer_manipulations as iman


def plInd1_to_plInd2(plInd1, l_p1_po1, l_p2_po1):
    """
    """
    uvec_p1 = plInd_to_uvecPO(plInd1, l_p1_po1)
    uvec_p2 = vecP1_to_vecP2(uvec_p1, l_p1_po1, l_p2_po1)
    plInd2 = vecP_to_plInd(uvec_p2, l_p2_po1)
    return plInd2


def plInd1_to_vecP2(plInd1, l_p1_po1, l_p2_po1):
    """
    """
    uvec_p1 = plInd_to_uvecP(plInd1, l_p1_po1)
    uvec_p2 = vecP1_to_vecP2(uvec_p1, l_p1_po1, l_p2_po1)
    return uvec_p2

def plInd1_to_vecPO2(plInd1, l_p1_po1, l_p2_po1):
    """
    """
    uvec_p1 = plInd_to_uvecP(plInd1, l_p1_po1)
    uvec_p2 = vecP1_to_vecP2(uvec_p1, l_p1_po1, l_p2_po1)
    uvec_po2 = l_p2_po.dot(uvec_p2)
    return uvec_p2

def vecP1_to_plInd2(vec_p1, l_p1_po1, l_p2_po1):
    """
    """
    plInd1 = vecP_to_plInd(vec_p1, l_p1_po1)
    plInd2 = plInd1_to_plInd2(plInd1, l_p1_po1, l_p2_po1)
    return plInd2

def vecP1_to_vecP2(vec_p1, l_p1_po1, l_p2_po1):
    """
    """
    l_po1_p2 = nla.inv(l_p2_po1)
    l_p1_p2 = l_po1_p2.dot(l_p1_po1)
    vec_p2 = l_p1_p2.dot(vec_p1)
    return vec_p2

def vecP1_to_vecPO2(vec_p1, l_p1_po1, l_p2_po1):
    """
    """
    vec_p2 = vecP1_to_vecP2(vec_p1, l_p1_po1, l_p2_po1)
    vec_po2 = vecP_to_vecPO(vec_p2, l_p2_po1)
    return vec_po2

def vecPO1_to_plInd2(vec_po1, l_p1_po1, l_p2_po1):
    """
    """
    vec_p1 = vecPO_to_vecP(vec_po1, l_p1_po1)
    plInd2 = vecP1_to_plInd2(vec_p1, l_p1_po1, l_p2_po1)
    return plInd2

def vecPO1_to_vecP2(vec_po1, l_p1_po1, l_p2_po1):
    """
    """
    vec_p1 = vecPO_to_vecP(vec_po1, l_p1_po1)
    vec_p2 = vecP1_to_vecP2(vec_p1, l_p1_po1, l_p2_po1)
    return vec_p2

def vecPO1_to_vecPO2(vec_po1, l_p_po1, l_p_po2):
    """
    """
    l_po1_po2 = l_p_po1.dot(nla.inv(l_p_po2))
    l_po2_po1 = nla.inv(l_po1_po2)
    vec_po2 = l_po1_po2.dot(vec_po1)

def vecP_to_vecPO(vec_p, l_p_po):
    """
    """
    vec_po = l_p_po.dot(vec_p)

def vecPO_to_vecP(vec_po, l_p_po):
    """
    """
    l_po_p = nla.inv(l_p_po)
    vec_p = l_po_p.dot(vec_p)


def vecP_to_plInd(uvec_p, l_p_po):
    """
    """
    uvec_po = l_p_po.dot(uvec_p)
    l_rp_po = fcd.reciprocal_mat(l_p_po)
    l_po_rp = nla.inv(l_rp_po)
    plInd1 = l_po_rp.dot(uvec_po)
    plInd, tm1 = iman.int_approx(plInd1)
    return plInd

def vecPO_to_plInd(uvec_po, l_p_po):
    """
    """
    l_rp_po = fcd.reciprocal_mat(l_p_po)
    l_po_rp = nla.inv(l_rp_po)
    plInd1 = l_po_rp.dot(uvec_po)
    plInd, tm1 = iman.int_approx(plInd1)
    return plInd

def plInd_to_uvecP(plInd, l_p_po):
    """
    """
    l_rp_po = fcd.reciprocal_mat(l_p_po)
    vec_po = l_rp_po.dot(plInd)
    uvec_po = vec_po/nla.norm(vec_po)
    l_po_p = nla.inv(l_p_po)
    uvec_p = l_po_p.dot(uvec_po)
    return uvec_p

def plInd_to_uvecPO(plInd, l_p_po):
    """
    """
    l_rp_po = fcd.reciprocal_mat(l_p_po)
    vec_po = l_rp_po.dot(plInd)
    uvec_po = vec_po/nla.norm(vec_po)
    return uvec_po





