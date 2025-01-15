import pandas as pd
import matplotlib.pyplot as plt

file_path = 'mesh.dat'
data = pd.read_csv(file_path, delimiter=' ', header=0)

print(data.head())

plt.scatter(data['X'], data['Y'])
plt.xlabel('X')
plt.ylabel('Y')


output_file = 'mesh_plot.png'
plt.savefig(output_file)

print('graph generated!')
