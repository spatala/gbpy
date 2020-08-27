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

        fig = plt.figure()
        plt.plot(s_file[:, 0], s_file[:, 1], 'o-', markeredgewidth=0, label="our code")
        plt.text(0, 1000, 'min_enery = ' + str(min_eng), fontsize=20)
        plt.xlabel('iteration', fontsize=20)
        plt.ylabel('energy', fontsize=20)
        plt.legend(['After min/heat/equil/cool/min'], fontsize=20)
        # plt.savefig('./results/fig/' + str(i) + '.jpg')
        plt.show()
        plt.close('all')
