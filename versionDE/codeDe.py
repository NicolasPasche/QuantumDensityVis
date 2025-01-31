import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.special import sph_harm

# Konstanten
hbar = 1.055e-34  # Reduziertes Plancksches Wirkungsquantum (Js)
m_e = 9.109e-31   # Elektronenmasse (kg)
a0 = 5.29e-11   # Bohrscher Radius (m) = hbar^2 / (m_e * e^2)

# Kugelkoordinaten-Raster
r = np.linspace(0, 10 * a0, 100)  # Radialabstand
theta = np.linspace(0, np.pi, 50)  # Polwinkel
phi = np.linspace(0, 2 * np.pi, 50)  # Azimutwinkel
R, Theta, Phi = np.meshgrid(r, theta, phi, indexing="ij")

# Kartesische Umrechnung für Visualisierung
X = R * np.sin(Theta) * np.cos(Phi)
Y = R * np.sin(Theta) * np.sin(Phi)
Z = R * np.cos(Theta)

# Radiale Funktion für 2p-Orbital (n=2, l=1) (Berechnet den radialen Anteil)
def radial_wave_function_2p(r):
    return (1 / (2 * np.sqrt(6) * a0**(3/2))) * (r / a0) * np.exp(-r / (2 * a0))

# Sphärische Harmonische für l=1, m=0
Y10 = sph_harm(0, 1, Phi, Theta)

# Wellenfunktion
psi_2p = radial_wave_function_2p(R) * Y10

# Wahrscheinlichkeitsdichte
prob_density = np.abs(psi_2p)**2

# Plot-Setup
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-5 * a0, 5 * a0)
ax.set_ylim(-5 * a0, 5 * a0)
ax.set_zlim(-5 * a0, 5 * a0)
ax.set_title("Wasserstoffatom: 2p-Orbital")

# Farben für die Wahrscheinlichkeitsdichte
colors = plt.cm.viridis(prob_density / np.max(prob_density))
scat = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=colors.reshape(-1, 4), s=0.1, alpha=0.8)

# Update-Funktion für Zeitabhängigkeit
def update(frame):
    time_factor = np.exp(-1j * frame * 0.05)  # Zeitabhängigkeit
    psi_t = psi_2p * time_factor
    prob_density_t = np.abs(psi_t)**2
    colors = plt.cm.viridis(prob_density_t / np.max(prob_density_t))
    scat.set_facecolor(colors.reshape(-1, 4))
    return scat,

# Animation
ani = FuncAnimation(fig, update, frames=600, interval=50, blit=True)

# Speichern der Animation als MP4
ani.save("wasserstoffatom_2p.mp4", writer="ffmpeg", fps=20)

plt.show()
