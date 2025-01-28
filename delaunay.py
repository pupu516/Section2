from collections import Counter
from scipy.spatial import Delaunay
import numpy as np


def surface1(coor):
    x = coor[0]
    y = coor[1]
    return 2*x**2 + 2*y**2

def surface2(coor):
    x = coor[0]
    y = coor[1]
    return 2*np.exp(-x**2 - y**2)

test_coor_set = np.empty((0, 2))
test_coor_set1 = test_coor_set2 = np.empty((0, 2))
density = 20
test_xcoor_set = np.linspace(-1, 1, density)
test_ycoor_set = np.linspace(-1, 1, density)

for x in test_xcoor_set:
    for y in test_ycoor_set:
        test_coor_set = np.append(test_coor_set, [np.array([x, y])], axis=0)

sample_cut = 1

for coor in test_coor_set:
    




