import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arrow

def r(t):
    return 2 + 12 * np.sin(t)

def phi(t):
    return t + 0.2 * np.cos(12 * t)

def polar_to_cartesian(t):
    r_t = r(t)
    phi_t = phi(t)
    x = r_t * np.cos(phi_t)
    y = r_t * np.sin(phi_t)
    return x, y

def velocity(t):
    dt = 0.0001
    x1, y1 = polar_to_cartesian(t)
    x2, y2 = polar_to_cartesian(t + dt)
    vx = (x2 - x1) / dt
    vy = (y2 - y1) / dt
    return vx, vy

def acceleration(t):
    dt = 0.0001
    vx1, vy1 = velocity(t)
    vx2, vy2 = velocity(t + dt)
    ax = (vx2 - vx1) / dt
    ay = (vy2 - vy1) / dt
    return ax, ay

def tangential_acceleration(t):
    dt = 0.0001
    vx1, vy1 = velocity(t)
    speed1 = np.sqrt(vx1**2 + vy1**2)
    vx2, vy2 = velocity(t + dt)
    speed2 = np.sqrt(vx2**2 + vy2**2)
    tang_acc = (speed2 - speed1) / dt
    speed = np.sqrt(vx1**2 + vy1**2)
    v_unit_x = vx1 / speed
    v_unit_y = vy1 / speed
    tang_acc_x = tang_acc * v_unit_x
    tang_acc_y = tang_acc * v_unit_y
    return tang_acc_x, tang_acc_y

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')

t_full = np.linspace(0, 10, 1000)
x_full, y_full = polar_to_cartesian(t_full)
ax.plot(x_full, y_full, 'b-', alpha=0.3, label='Траектория')

point, = ax.plot([], [], 'ro', markersize=10, label='Точка')
velocity_arrow = ax.quiver([], [], [], [], color='g', scale=200, label='Вектор скорости')
acceleration_arrow = ax.quiver([], [], [], [], color='r', scale=1000, label='Вектор ускорения')
tangential_arrow = ax.quiver([], [], [], [], color='blue', scale=1000, label='Тангенциальное ускорение')

velocity_proxy = plt.Rectangle((0, 0), 1, 1, fc='g', label='Вектор скорости')
acceleration_proxy = plt.Rectangle((0, 0), 1, 1, fc='r', label='Вектор ускорения')
tangential_proxy = plt.Rectangle((0, 0), 1, 1, fc='blue', label='Тангенциальное ускорение')

margin = 2
ax.set_xlim(np.min(x_full) - margin, np.max(x_full) + margin)
ax.set_ylim(np.min(y_full) - margin, np.max(y_full) + margin)
ax.grid(True)

ax.legend([point, velocity_proxy, acceleration_proxy, tangential_proxy], 
         ['Точка', 'Вектор скорости', 'Вектор ускорения', 'Тангенциальное ускорение'],
         loc='upper right')

def animate(frame):
    t = frame * 0.05
    x, y = polar_to_cartesian(t)
    vx, vy = velocity(t)
    ax, ay = acceleration(t)
    tax, tay = tangential_acceleration(t)
    point.set_data([x], [y])
    velocity_arrow.set_offsets([x, y])
    velocity_arrow.set_UVC([vx], [vy])
    acceleration_arrow.set_offsets([x, y])
    acceleration_arrow.set_UVC([ax], [ay])
    tangential_arrow.set_offsets([x, y])
    tangential_arrow.set_UVC([tax], [tay])
    return point, velocity_arrow, acceleration_arrow, tangential_arrow

anim = FuncAnimation(fig, animate, frames=200, interval=100, blit=True)

plt.title('Движение точки по траектории')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()