#!/bin/bash

import gbpy.byxtal.lll_tools as lt
import numpy as np

int_mat = np.array([[1,-1,3],[1,0,5],[1,2,6]])
print(lt.lll_reduction(int_mat))