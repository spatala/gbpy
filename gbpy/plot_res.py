import numpy as np
import matplotlib.pyplot as plt
import os
import re
p_leila = './results/'


# directory = [p_leila, p_arash]
directory = [p_leila]
j = 0
for dir0 in directory:
    min_eng = []
    n_accept = []
    a = os.listdir(dir0)
    for i in a:
        print(a)
        file0 = np.loadtxt(dir0 + i)
        s_file = file0[np.argsort(file0[:, 0])]
        min_eng = min_eng + [np.min(s_file[:, 1])]
        n_accept = n_accept + [np.shape(s_file)[0]]

        fig = plt.figure(figsize=(25, 5))
        plt.scatter(s_file[:, 0], s_file[:, 1],  color = "darkcyan", marker="*")
        plt.text(2500, 370, 'min_enery = ' + str(min_eng)[1:8] + ' $mJ/m^2$', fontsize=18)
        # plt.ylim(355, 600)
        plt.xlim(0,5100)
        plt.tick_params(axis='both', labelsize=18)
        plt.xlabel('iteration', fontsize=18)
        plt.ylabel('energy($mJ/m^2$)', fontsize=18)
        # plt.xticks(fontsize= 10)
        plt.legend(['gb_attr_Mg_sun_S13_1_N1_0_0_1_N2_0_0_-1'], fontsize=20)
        # plt.savefig('./results/fig/' + str(i) + '.jpg')
        plt.show()
        plt.close('all')

# plt.hist(s_file[:, 1], bins=50, color="darkcyan")
# plt.ylim(0, 500)
# plt.xlim(355,580)
# plt.xlabel('energy($mJ/m^2$)', fontsize=24)
# plt.ylabel('frequency', fontsize=24)
# plt.tick_params(axis='both', labelsize=24)
# plt.show()