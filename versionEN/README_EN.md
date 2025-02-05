# Visualization of the 2p orbital of a hydrogen atom

This project demonstrates the visualization of the 2p orbital of a hydrogen atom using the Schrödinger equation, 
spherical harmonic functions and probability densities. The simulation is created with Python and shows 
the evolution of the orbital over time.

## Mathematics Behind the Code

The hydrogen atom is described by the Schrödinger equation. For the visualization of the orbital, we 
need two essential mathematical components:

### 1. Radial part of the wave function

The radial function for the 2p orbital $$n=2$$, $$l=1$$ is:

$$
R_{2p}(r) = \frac{1}{2\sqrt{6} \, a_0^{3/2}} \cdot \frac{r}{a_0} \cdot e^{-r / (2a_0)}
$$

Where:
- $$a_0$$: Bohr radius $$(a_0 = 5.29 \times 10^{-11} \, \text{m})$$
- $$r$$: Radial distance to the core

### 2. Spherical harmonics

The spherical harmonic functions $$Y_l^m(\theta, \phi)$$ describe the angular part of the wave function.
For $$l=1$$, $$m=0$$, we have:

$$
Y_1^0(\theta, \phi) = \sqrt{\frac{3}{4\pi}} \cos(\theta)
$$

### 3. Total wave function

The toal wave function comines the radial and angular parts:

$$
\Psi_{2p}(r, \theta, \phi) = R_{2p}(r) \cdot Y_1^0(\theta, \phi)
$$

The probability density is given by the square of the magnitude of the wave function:

$$
|\Psi_{2p}(r, \theta, \phi)|^2 = \left| R_{2p}(r) \cdot Y_1^0(\theta, \phi) \right|^2
$$

## Python Code for the Implementation

### Importing the required libraries
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.special import sph_harm
```

### Defining the constants
```python
# Constants
hbar = 1.055e-34  # Reduced Planck's constant (Js)
m_e = 9.109e-31   # Electron mass (kg)
a0 = 5.29e-11     # Bohr radius (m) = hbar^2 / (m_e * elementary charge^2)
```

### Creating the spherical coordinate grid
```python
# Spherical coordinate grid
r = np.linspace(0, 10 * a0, 100)  # Radial distance
theta = np.linspace(0, np.pi, 50)  # Polar angle
phi = np.linspace(0, 2 * np.pi, 50)  # Azimuthal angle
R, Theta, Phi = np.meshgrid(r, theta, phi, indexing="ij")
```

### Visualization
```python
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
```

### Animation of the Evolution Over Time
```python
# Update function for time dependence
def update(frame):
    time_factor = np.exp(-1j * frame * 0.05)  # Time dependence
    psi_t = psi_2p * time_factor
    prob_density_t = np.abs(psi_t)**2
    colors = plt.cm.viridis(prob_density_t / np.max(prob_density_t))
    scat.set_facecolor(colors.reshape(-1, 4))
    return scat,
```

The animation is created with `FuncAnimation`, saved and a preview is plotted
```python
# Animation
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# Saving the animation as MP4
ani.save("hydrogen_atom_2p.mp4", writer="ffmpeg", fps=20)

plt.show()
```

## Final Product
![image](https://github.com/user-attachments/assets/a6659d86-face-42a1-9fd3-04e4cecd2825)

