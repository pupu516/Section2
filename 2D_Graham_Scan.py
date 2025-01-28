import numpy as np
import matplotlib.pyplot as plt

file_path = 'mesh.dat'
data = np.loadtxt(file_path, skiprows=1)


index_of_miny = np.argmin(data.T[1])
number = len(data)
miny_coor = data[index_of_miny]

def polar_angle(arr1, arr2):
    delta_x = arr2[0] - arr1[0]
    r = np.sqrt((arr2[0] - arr1[0])**2 + (arr2[1] - arr1[1])**2)
    return np.arccos(delta_x/r)

#data_with_angle = np.append(data.T, [np.zeros(len(data.T[1]))], axis=0).T

#print(data_with_angle)

datacp = data

data_order = np.array([datacp[index_of_miny]])
datacp = np.delete(datacp, index_of_miny, axis=0)

min_angle = 0 
min_index = 0

for i in range(number - 2):
    min_angle = 1000
    for j in range(number - 1 - i):
        angle = polar_angle(datacp[j], miny_coor) 
        if (angle < min_angle):
            min_index = j
            min_angle = angle
    data_order = np.append(data_order, [datacp[min_index]], axis=0)
    datacp = np.delete(datacp, min_index, axis=0)

data_order = np.append(data_order, datacp, axis=0)

#print(data_order)

'''
a = np.array([])
for coor in data_order:
    angle = polar_angle(miny_coor, coor)
    a = np.append(a, angle)
b = np.append(data_order.T, [a], axis=0).T
print(b)
'''
    
def solve_intersection(p1, p2, p3, p4):
    m1 = (p2[1] - p1[1])/(p2[0] - p1[0])
    b1 = p1[1] - p1[0] * m1

    m2 = (p4[1] - p3[1])/(p4[0] - p3[0])
    b2 = p3[1] - p3[0] * m2

    A = np.array([[-m1, 1], [-m2, 1]])
    B = np.array([b1, b2])

    intersection = np.linalg.solve(A, B)
    return intersection



def save_or_not(miny_coor, coor1, coor2, coor3):
    intersect = solve_intersection(miny_coor, coor2, coor1, coor3)
    nocon_distance = np.sqrt((miny_coor[0] - intersect[0])**2 + (miny_coor[1] - intersect[1])**2)
    actual_distance = np.sqrt((miny_coor[0] - coor2[0])**2 + (miny_coor[1] - coor2[1])**2)
    return actual_distance > nocon_distance



current_index = 2
data_finalcut = data_order
loopround = 0

while (current_index != len(data_finalcut)):
    loopround+=1
    save = save_or_not(miny_coor, data_finalcut[current_index - 2], data_finalcut[current_index -1], data_finalcut[current_index - 0])
    if save:
        current_index += 1
    else:
        print('im here')
        data_finalcut = np.delete(data_finalcut, current_index - 1, axis=0)
        current_index -= 1
        

print(data_finalcut)
print(len(data_order))
print(len(data_finalcut))
print(loopround)


x_data = data[:, 0]
y_data = data[:, 1]

x_cut = data_finalcut[:, 0]
y_cut = data_finalcut[:, 1]
plt.plot(x_cut, y_cut)
plt.scatter(x_data, y_data)
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('2d_graham_scan.png')
print('graph generated')




