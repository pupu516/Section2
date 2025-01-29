import numpy as np
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Define the functions
def surface1(x, y):
    return 2 * x**2 + 2 * y**2

def surface2(x, y):
    return 2 * np.exp(-x**2 - y**2)

# Part a: Generate Surface Point Clouds
def generate_surface_point_clouds(resolution=50):
    x = np.linspace(-2, 2, resolution)
    y = np.linspace(-2, 2, resolution)
    X, Y = np.meshgrid(x, y)
    X_flat, Y_flat = X.ravel(), Y.ravel()
    
    # Compute Z for both surfaces
    Z1 = surface1(X_flat, Y_flat)
    Z2 = surface2(X_flat, Y_flat)
    
    # Combine into point clouds
    top_surface = np.column_stack((X_flat, Y_flat, Z1))
    bottom_surface = np.column_stack((X_flat, Y_flat, Z2))
    
    return top_surface, bottom_surface

# Part b: Delaunay Triangulation for Surface Mesh
def generate_surface_mesh(top_surface, bottom_surface):
    # Perform Delaunay triangulation on the X, Y coordinates
    tri_top = Delaunay(top_surface[:, :2])
    tri_bottom = Delaunay(bottom_surface[:, :2])
    
    # Adjust vertex indices for bottom surface to ensure unique IDs
    bottom_surface_shifted = bottom_surface + [0, 0, 0]
    tri_bottom_shifted = tri_bottom.simplices + len(top_surface)
    
    # Combine top and bottom surfaces
    combined_points = np.vstack((top_surface, bottom_surface_shifted))
    combined_triangulation = np.vstack((tri_top.simplices, tri_bottom_shifted))
    
    return combined_points, combined_triangulation

def visualize_surface_mesh(points, simplices, title="Surface Mesh"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=simplices, cmap="viridis", alpha=0.8)
    ax.set_title(title)
    plt.show()

# Part c: Delaunay Triangulation for Volume Mesh
def generate_volume_mesh(top_surface, bottom_surface, resolution=50):
    # Create a grid of points between top and bottom surfaces
    z_levels = np.linspace(0, 1, resolution)
    volume_points = []
    for z in z_levels:
        interpolated_surface = (1 - z) * top_surface + z * bottom_surface
        volume_points.append(interpolated_surface)
    volume_points = np.vstack(volume_points)
    
    # Perform Delaunay triangulation in 3D
    tri_volume = Delaunay(volume_points)
    
    return volume_points, tri_volume

def visualize_volume_mesh(points, simplices, title="Volume Mesh"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for simplex in simplices.simplices:
        vertices = points[simplex]
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], color="cyan", alpha=0.2)
    ax.set_title(title)
    plt.show()

# Part d: Extract and Visualize Surface Mesh from Volume Mesh
def extract_surface_from_volume(points, simplices):
    # Find boundary triangles
    boundary_faces = {}
    for simplex in simplices.simplices:
        for i in range(4):
            face = tuple(sorted(simplex[np.arange(4) != i]))
            if face in boundary_faces:
                boundary_faces[face] += 1
            else:
                boundary_faces[face] = 1
    
    # Keep only boundary faces (appear once)
    boundary_faces = [face for face, count in boundary_faces.items() if count == 1]
    surface_simplices = np.array(boundary_faces)
    
    return surface_simplices

def main():
    # Part a: Generate Surface Point Clouds
    top_surface, bottom_surface = generate_surface_point_clouds(resolution=50)
    
    # Part b: Generate and Visualize Surface Mesh
    combined_points, combined_triangulation = generate_surface_mesh(top_surface, bottom_surface)
    visualize_surface_mesh(combined_points, combined_triangulation, title="Combined Surface Mesh")
    
    # Part c: Generate and Visualize Volume Mesh
    volume_points, tri_volume = generate_volume_mesh(top_surface, bottom_surface, resolution=10)
    visualize_volume_mesh(volume_points, tri_volume, title="Volume Mesh")
    
    # Part d: Extract and Visualize Surface Mesh from Volume Mesh
    surface_simplices = extract_surface_from_volume(volume_points, tri_volume)
    visualize_surface_mesh(volume_points, surface_simplices, title="Surface Extracted from Volume Mesh")

# Run the main function
if __name__ == "__main__":
    main()

