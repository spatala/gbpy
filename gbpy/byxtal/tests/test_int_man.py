import gbpy.byxtal.integer_manipulations as iman;
import numpy as np
import pickle as pkl

import os
import inspect
import gbpy

byxtal_dir = os.path.dirname((inspect.getfile(gbpy.byxtal)))

## Directory and file names
pkl_dir = byxtal_dir+'/tests/pkl_files/'
##############################################################


####################################################
tau = 3/2.0
tol1 = 1e-6
n1, d1 = iman.rat_approx(np.array(tau), tol1);
print(np.abs(tau - n1/d1))
####################################################

####################################################
a1 = 3/np.sqrt(19)
b1 = 2/np.sqrt(19)
c1 = -5/np.sqrt(19)
iarr1 = np.array([a1, b1, c1])
iarr2, m1 = iman.int_approx(iarr1)
print(np.linalg.norm(iarr1*m1 - iarr2))
####################################################

####################################################
tol1 = 1e-6
ct3 = 0
num1 = 100
diff_arr = np.zeros((num1,1))
while ct3 < num1:
    rMat = np.random.rand(3,2)
    N1, D1 = iman.rat_approx(rMat, tol1)
    diff_mat = 0*rMat
    sz = np.shape(rMat)
    for ct1 in range(sz[0]):
        for ct2 in range(sz[1]):
            diff_mat[ct1,ct2] = rMat[ct1,ct2] - N1[ct1, ct2]/D1[ct1, ct2];
    diff_arr[ct3] = np.max(np.abs(diff_mat));
    ct3 = ct3 + 1;

print(np.max(diff_arr))
####################################################

####################################################
r1 = np.random.rand(8,2,6,8);
N1, D1 = iman.rat_approx(r1)
print(np.max(np.abs(r1 - N1/D1)))
####################################################

####################################################
r1 = np.random.rand(5,8);
N1, D1 = iman.rat_approx(r1)
print(np.max(np.abs(r1 - N1/D1)))
####################################################

####################################################
r1 = np.random.rand(5,1);
N1, D1 = iman.rat_approx(r1)
print(np.max(np.abs(r1 - N1/D1)))
####################################################

####################################################
r1 = np.random.rand(5);
N1, D1 = iman.rat_approx(r1)
print(np.max(np.abs(r1 - N1/D1)))
####################################################

####################################################
r1 = np.random.rand(1,5);
N1, D1 = iman.rat_approx(r1)
lcm1 = iman.lcm_array(D1)
print(np.max(np.abs(r1 - N1/D1)))
####################################################

pkl_name = pkl_dir+'vecs.pkl'
jar = open(pkl_name,'rb')
vecs_dict = pkl.load(jar)
jar.close()

vecs = vecs_dict['vecs']

sz1 = np.shape(vecs)[0]

diff_n = np.zeros((sz1,))
for ct1 in range(sz1):
    print(ct1)
    vec1 = vecs[ct1]
    u_vec1 = vec1/np.linalg.norm(vec1)
    i1, m1 = iman.int_approx(u_vec1, 1e-6)

    d_vec = i1-vec1
    if np.linalg.norm(d_vec) > 1e-10:
        ind1 = np.where(vec1 != 0)[0]
        m_arr = (np.unique(vec1[ind1]/i1[ind1]))
        m2 = m_arr[0]
        diff_n[ct1] = np.linalg.norm(i1*m2 - vec1)
    else:
        diff_n[ct1] = np.linalg.norm(vec1-i1)


