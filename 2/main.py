import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Arc, Rectangle

R = 2.0
r = 1.0
L = 1.5
M1 = 1.0
M2 = 0.5
c = 1.0

t = np.linspace(0, 10, 200)
omega = 1.0

def update(frame):
    plt.clf()
    
    phi = omega * t[frame]
    theta = -phi * R/r
    
    xA = L * np.cos(phi)
    yA = L * np.sin(phi)
    
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.gca().set_aspect('equal')
    plt.grid(True)
    
    circle1 = plt.Circle((0, 0), R, fill=False, color='blue')
    plt.gca().add_patch(circle1)
    
    circle2 = plt.Circle((xA, yA), r, fill=False, color='red')
    plt.gca().add_patch(circle2)
    
    plt.plot([0, xA], [0, yA], 'k-', linewidth=2)
    
    plt.plot(0, 0, 'ko', markersize=10)
    plt.plot(xA, yA, 'ko', markersize=10)
    
    plt.text(-0.2, -0.2, 'O', fontsize=12)
    plt.text(xA-0.2, yA-0.2, 'A', fontsize=12)
    
    plt.plot([0, R*np.cos(phi)], [0, R*np.sin(phi)], 'b--')
    plt.plot([xA, xA + r*np.cos(theta)], [yA, yA + r*np.sin(theta)], 'r--')
    
    plt.title(f't = {t[frame]:.2f} s')

fig = plt.figure(figsize=(8, 8))

anim = FuncAnimation(fig, update, frames=len(t), interval=50, repeat=True)

plt.show()