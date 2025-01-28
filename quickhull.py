import numpy as np
import matplotlib.pyplot as plt

file_path = 'mesh.dat'
data = np.loadtxt(file_path, skiprows=1)


# each step, 

def P_finder(coor1, coor2, data_arr):
    
    P_candidate = 0
    distance = 0
    vec1 = coor2 - coor1
    norm = np.linalg.norm(vec1)
    for i in len(data_arr):
        coor = data_arr[i]
        vec2 = coor - coor1
        cross_product = np.cross(vec1, vec2)
        dist_temp = cross_product / norm
        if dist_temp > distance:
            distance = dist_temp
            P_candidate = i
    
    return P_candidate

def left_side_check(coor1, coor2, coor):
    vec1 = coor2 - coor1 
    vec2 = coor - coor1
    cross_product = np.cross(vec1, vec2)
    return (cross_product > 0)

def inside_triangle(coor1, coor2, coor3, coor):
    check1 = left_side_check(coor1, coor2, coor)
    check2 = left_side_check(coor2, coor3, coor)
    check3 = left_side_check(coor3, coor1, coor)

    return (check1 and check2 and check3)

def categorizer(coor1, coor2, coor3, data_arr):
    pass 


# starting from two points, one data set, ending in two points, one data set

def iterator(coor1, coor2, dataset):
    dataset = dataset
    P = P_finder(coor1, coor2, dataset)
    P_coor = dataset[P]
    dataset = np.delete(dataset, P)
    for i in range(len(dataset)):
        coor = dataset[i]
        if inside_triangle(coor1, coor2, P_coor, coor):
            dataset = np.delete(dataset, i)
    newarr1 = np.array([])
    newarr2 = np.array([])

    for i in range(len(dataset)):
        if left_side_check(coor2, P_coor, dataset[i]):
            newarr1 = np.append(newarr1, [dataset[i]], axis=0)
        else:
            newarr2 = np.append(newarr2, [dataset[i]], axis=0)
    return P_coor, coor1, coor2, newarr1, newarr2


datacp = data

data_envelop = np.array([])

iterate_stuff = []

index_minx_coor = np.argmin(data.T[0])
index_maxx_coor = np.argmax(data.T[0])

data_envelop = np.array([datacp[index_minx_coor]])
data_envelop = np.append(data_envelop, [datacp[index_maxx_coor]], axis=0)

datacp = np.delete(datacp, [index_minx_coor, index_maxx_coor], axis=0)

arr1 = np.array([])
arr2 = np.array([])

for i in range(len(datacp)):
    if left_side_check(data_envelop[0], data_envelop[1], datacp[i]):
        arr1 = np.append(arr1, [datacp[i]], axis=0)
    else:
        arr2 = np.append(arr2, [datacp[i]], axis=0)

ls1 = [data_envelop[0], data_envelop[1], arr1, False]
ls2 = [data_envelop[1], data_envelop[0], arr2, False]

iteratee = [ls1, ls2]

while True:
    for ite in iteratee:
        if not ite[3]:
           P_coor, coor1, coor2, newarr1, newarr2 = iterator(ite[0], ite[1], ite[2]) 
           ite[3] = True
           data_envelop = np.append(data_envelop, [P_coor], axis=0)
           iteratee.append([coor1, P_coor, newarr1, False])
           iteratee.append([P_coor, coor2, newarr2, False])

    coor_left = 0
    for ite in iteratee:
        if ite[3]:
            continue
        else:
            coor_left += len(ite[2])
    
    if (coor_left == 0):
        break

print(data_envelop)


            
            
            
    






