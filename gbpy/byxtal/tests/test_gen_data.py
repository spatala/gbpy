import numpy as np

num1 = 10
num2 = 2*num1+1
vecs = np.zeros((num2*num2*num2,3))
ct4 = 0
for ct1 in range(-num1,num1+1):
    print(ct1)
    for ct2 in range(-num1,num1+1):
        for ct3 in range(-num1,num1+1):
            vecs[ct4,:] = [ct1, ct2, ct3]
            ct4 = ct4 + 1


ind1 = np.where(np.sqrt(np.sum(vecs**2, axis=1)) == 0)[0][0]
vecs = np.delete(vecs, ind1, axis=0)


num1 = 10000
vecs1 = np.zeros((num1,3))
for ct1 in range(num1):
    tct1 = np.random.randint(-num1, num1)
    tct2 = np.random.randint(-num1, num1)
    tct3 = np.random.randint(-num1, num1)
    vecs1[ct1] = [tct1, tct2, tct3]

vecs = np.vstack((vecs1, vecs))

import pickle as pkl
pkl_name = 'vecs.pkl'
jar = open(pkl_name,'wb')
vecs_dict = {}
vecs_dict['vecs'] = vecs
pkl.dump(vecs_dict, jar)
jar.close()

