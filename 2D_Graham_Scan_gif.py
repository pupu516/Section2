import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load data
file_path = 'mesh.dat'
data = np.loadtxt(file_path, skiprows=1)

# Find the starting point (minimum y-coordinate)
index_of_miny = np.argmin(data.T[1])
miny_coor = data[index_of_miny]

datacp = data
data_order = np.array([datacp[index_of_miny]])
datacp = np.delete(datacp, index_of_miny, axis=0)

# Sort points by polar angle
for i in range(len(datacp)):
    min_angle = float('inf')
    min_index = 0
    for j in range(len(datacp)):
        angle = np.arctan2(datacp[j][1] - miny_coor[1], datacp[j][0] - miny_coor[0])
        if angle < min_angle:
            min_angle = angle
            min_index = j
    data_order = np.append(data_order, [datacp[min_index]], axis=0)
    datacp = np.delete(datacp, min_index, axis=0)

# Close the loop by appending the first point
data_order = np.append(data_order, [data_order[0]], axis=0)

# Function to check convexity
def save_or_not(p1, p2, p3):
    dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
    dx2, dy2 = p3[0] - p2[0], p3[1] - p2[1]
    cross_product = dx1 * dy2 - dy1 * dx2  # Cross product
    return cross_product > 0

# Generate the convex hull process step by step
steps = []
data_finalcut = data_order[:2].tolist()  # Start with the first two points
steps.append(np.array(data_finalcut))

for i in range(2, len(data_order)):
    data_finalcut.append(data_order[i])
    while len(data_finalcut) > 2 and not save_or_not(data_finalcut[-3], data_finalcut[-2], data_finalcut[-1]):
        data_finalcut.pop(-2)  # Remove the second-to-last point
    steps.append(np.array(data_finalcut))  # Save the current state

# Animation setup
fig, ax = plt.subplots()
ax.set_xlim(0, np.max(data[:, 0]) + 1)
ax.set_ylim(0, np.max(data[:, 1]) + 1)
ax.set_title("Graham Scan Convex Hull Process")
ax.set_xlabel("X")
ax.set_ylabel("Y")
line, = ax.plot([], [], 'b-', label="Convex Hull")
points, = ax.plot(data[:, 0], data[:, 1], 'ro', label="Data Points")
current_point, = ax.plot([], [], 'go', label="Current Points")

# Animation initialization
def init():
    line.set_data([], [])
    current_point.set_data([], [])
    return line, current_point

# Animation update function
def update(frame):
    step_data = steps[frame]
    line.set_data(step_data[:, 0], step_data[:, 1])  # Update the hull
    current_point.set_data(step_data[:, 0], step_data[:, 1])  # Update current points
    return line, current_point

# Create the animation
anim = FuncAnimation(fig, update, frames=len(steps), init_func=init, blit=True)

# Save as GIF
anim.save('graham_scan_process.gif', writer='imagemagick', fps=2)
print("GIF saved as 'graham_scan_process.gif'")

plt.legend()
plt.show()

