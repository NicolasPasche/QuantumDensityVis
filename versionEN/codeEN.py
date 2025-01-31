import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.special import sph_harm

# Constants
hbar = 1.055e-34  # Reduced Planck's constant (Js)
m_e = 9.109e-31   # Electron mass (kg)
a0 = 5.29e-11     # Bohr radius (m) = hbar^2 / (m_e * elementary charge^2)

# Spherical coordinate grid
r = np.linspace(0, 10 * a0, 100)  # Radial distance
theta = np.linspace(0, np.pi, 50)  # Polar angle
phi = np.linspace(0, 2 * np.pi, 50)  # Azimuthal angle
R, Theta, Phi = np.meshgrid(r, theta, phi, indexing="ij")

def radial_wave_function_2p(r):
    # Radial part of the wave function for the 2p orbital
    return (1 / (2 * np.sqrt(6) * a0**(3/2))) * (r / a0) * np.exp(-r / (2 * a0))

# Spherical harmonics for l=1, m=0
Y10 = sph_harm(0, 1, Phi, Theta)

# Wave function
psi_2p = radial_wave_function_2p(R) * Y10

# Probability density
prob_density = np.abs(psi_2p)**2

# Plot setup
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-5 * a0, 5 * a0)
ax.set_ylim(-5 * a0, 5 * a0)
ax.set_zlim(-5 * a0, 5 * a0)
ax.set_title("Hydrogen Atom: 2p Orbital")

# Colors for the probability density
colors = plt.cm.viridis(prob_density / np.max(prob_density))
scat = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=colors.reshape(-1, 4), s=0.1, alpha=0.8)

# Update function for time dependence
def update(frame):
    time_factor = np.exp(-1j * frame * 0.05)  # Time dependence
    psi_t = psi_2p * time_factor
    prob_density_t = np.abs(psi_t)**2
    colors = plt.cm.viridis(prob_density_t / np.max(prob_density_t))
    scat.set_facecolor(colors.reshape(-1, 4))
    return scat,

# Animation
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# Saving the animation as MP4
ani.save("hydrogen_atom_2p.mp4", writer="ffmpeg", fps=20)

plt.show()
