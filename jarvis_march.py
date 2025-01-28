import numpy as np
import matplotlib.pyplot as plt

file_path = 'mesh.dat'
data = np.loadtxt(file_path, skiprows=1)

index_of_minx = np.argmin(data.T[0])
number = len(data)
minx_coor = data[index_of_minx]

def polar_angle(coor1, coor2, coor3):
    vec1 = coor2 - coor1
    vec2 = coor3 - coor1
    cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
#    angle = np.arccos(cosine)
    return cosine

assist_coor = minx_coor + np.array([0, 1])

datacp = np.concatenate([[assist_coor], [minx_coor], np.delete(data, index_of_minx, axis=0)], axis=0)

data_envelop = np.array([minx_coor])


current_index = 1
min_angle = 0
min_angle_index = 0

coor1 = datacp[0]
coor2 = datacp[1]

while True:

    min_angle = -100
    
    next_coor = np.array([])
    for i in range(len(datacp)):
        coor = datacp[i]
        if (np.array_equal(coor, coor1) or np.array_equal(coor, coor2) or np.array_equal(coor, assist_coor)):
            continue
        angle = polar_angle(coor1, coor2, coor)
        if (angle > min_angle):
            min_angle = angle
            next_coor = datacp[i]
    if np.array_equal(next_coor, datacp[1]):
        break

    data_envelop = np.append(data_envelop, [next_coor], axis=0)  
    coor1 = coor2
    coor2 = next_coor

print(data_envelop)
     






