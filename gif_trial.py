import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simplified figure and axis setup
fig, ax = plt.subplots()
x = np.linspace(0, 2 * np.pi, 50)  # Reduced points for lower memory usage
line, = ax.plot(x, np.sin(x))

# Axis limits
ax.set_ylim(-1.2, 1.2)

# Animation function
def update(frame):
    line.set_ydata(np.sin(x + frame / 5.0))
    return line,

# Create lightweight animation
anim = FuncAnimation(fig, update, frames=20, interval=200, blit=True)

# Save as GIF
anim.save('optimized_sine_wave.gif', writer='imagemagick', fps=5)
print("Lightweight GIF saved as 'optimized_sine_wave.gif'")

