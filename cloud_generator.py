import numpy as np

def generate_point_cloud(n, x_bounds=(0, 1), y_bounds=(0, 1), random_seed=None):

    x = np.random.uniform(x_bounds[0], x_bounds[1], n)
    y = np.random.uniform(y_bounds[0], y_bounds[1], n)
    
    return np.column_stack((x, y))

point_cloud = generate_point_cloud(30, random_seed=42)
print(point_cloud)

