import numpy as np
import time
import matplotlib.pyplot as plt

# -------------------------
# Graham Scan Algorithm
# -------------------------
def graham_scan(points):
    points = sorted(points, key=lambda p: (p[0], p[1]))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]

# -------------------------
# Jarvis March Algorithm
# -------------------------
def jarvis_march(points):
    points = np.array(points)
    n = len(points)
    if n < 3:
        return points

    leftmost_idx = np.argmin(points[:, 0])
    hull = []
    point_idx = leftmost_idx

    while True:
        hull.append(points[point_idx])
        next_idx = (point_idx + 1) % n
        for i in range(n):
            if i != point_idx:
                cross = np.cross(points[next_idx] - points[point_idx], points[i] - points[point_idx])
                if cross < 0 or (cross == 0 and np.linalg.norm(points[next_idx] - points[point_idx]) <
                                 np.linalg.norm(points[i] - points[point_idx])):
                    next_idx = i

        point_idx = next_idx
        if point_idx == leftmost_idx:
            break

    return np.array(hull)

# -------------------------
# Quickhull Algorithm
# -------------------------
def quickhull(points):
    points = np.array(points)

    def farthest_point(p1, p2, points):
        distances = np.abs(np.cross(p2 - p1, points - p1)) / np.linalg.norm(p2 - p1)
        farthest_idx = np.argmax(distances)
        return points[farthest_idx]

    def find_hull(points, p1, p2):
        if len(points) == 0:
            return []
        farthest = farthest_point(p1, p2, points)
        left_of_p1 = points[np.cross(farthest - p1, points - p1) > 0]
        left_of_p2 = points[np.cross(p2 - farthest, points - farthest) > 0]

        return find_hull(left_of_p1, p1, farthest) + [farthest] + find_hull(left_of_p2, farthest, p2)

    leftmost = points[np.argmin(points[:, 0])]
    rightmost = points[np.argmax(points[:, 0])]

    upper_set = points[np.cross(rightmost - leftmost, points - leftmost) > 0]
    lower_set = points[np.cross(leftmost - rightmost, points - rightmost) > 0]

    upper_hull = find_hull(upper_set, leftmost, rightmost)
    lower_hull = find_hull(lower_set, rightmost, leftmost)

    return [leftmost] + upper_hull + [rightmost] + lower_hull

# -------------------------
# Monotone Chain Algorithm
# -------------------------
def monotone_chain(points):
    points = sorted(points, key=lambda p: (p[0], p[1]))

    lower = []
    for p in points:
        while len(lower) >= 2 and np.cross(lower[-1] - lower[-2], p - lower[-2]) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and np.cross(upper[-1] - upper[-2], p - upper[-2]) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]

# -------------------------
# Generate Point Cloud
# -------------------------
def generate_point_cloud(n, x_bounds=(0, 1), y_bounds=(0, 1), random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    x = np.random.uniform(x_bounds[0], x_bounds[1], n)
    y = np.random.uniform(y_bounds[0], y_bounds[1], n)
    
    return np.column_stack((x, y))

# -------------------------
# Measure runtime for all four algorithms
# -------------------------
n_values = [10, 50, 100, 200, 400, 800, 1000]
runtime_graham_scan = []
runtime_jarvis_march = []
runtime_quickhull = []
runtime_monotone_chain = []

for n in n_values:
    point_cloud = generate_point_cloud(n, random_seed=42)
    
    start_time = time.time()
    graham_scan(point_cloud)
    runtime_graham_scan.append(time.time() - start_time)
    
    start_time = time.time()
    jarvis_march(point_cloud)
    runtime_jarvis_march.append(time.time() - start_time)
    
    start_time = time.time()
    quickhull(point_cloud)
    runtime_quickhull.append(time.time() - start_time)
    
    start_time = time.time()
    monotone_chain(point_cloud)
    runtime_monotone_chain.append(time.time() - start_time)

# -------------------------
# Plot runtime comparison
# -------------------------
plt.figure(figsize=(12, 8))
plt.plot(n_values, runtime_graham_scan, label="Graham Scan (O(n log n))", marker='o')
plt.plot(n_values, runtime_jarvis_march, label="Jarvis March (O(nh))", marker='o')
plt.plot(n_values, runtime_quickhull, label="Quickhull (O(n log n))", marker='o')
plt.plot(n_values, runtime_monotone_chain, label="Monotone Chain (O(n log n))", marker='o')

plt.xlabel("Number of Points (n)")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime Comparison of Convex Hull Algorithms")
plt.legend()
plt.grid(True)
plt.savefig("graph.png")


