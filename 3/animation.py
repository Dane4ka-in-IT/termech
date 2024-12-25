import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Arrow, Wedge

m1 = 2.0
m2 = 1.0
m3 = 1.0
R = 1.0
r = 0.2
c = 0.5
M1 = 2.0
M2 = 1.0
g = 9.81

theta0 = np.pi/2
theta_dot0 = np.pi/2
phi0 = 0.0
phi_dot0 = np.pi/4

t = np.linspace(0, 5, 250)

def equations_of_motion(state, t):
    theta, theta_dot, phi, phi_dot = state
    
    a11 = (m1 + m2)*R**2 + m3*(R - r)**2
    a12 = m2*R
    a21 = m2*R
    a22 = (3*m2 + (2/3)*m3)*(R - r)
    
    b1 = 2*M1
    b2 = 2*c*phi + 2*M2 - (2*m2 + m3)*g*np.sin(phi)*(R - r)
    
    det = a11*a22 - a12*a21
    theta_ddot = (b1*a22 - b2*a12)/det
    phi_ddot = (-b1*a21 + b2*a11)/det
    
    return [theta_dot, theta_ddot, phi_dot, phi_ddot]

def draw_gear(center_x, center_y, radius, angle, num_teeth=20, color='blue'):
    gear = plt.Circle((center_x, center_y), radius, fill=False, color=color)
    plt.gca().add_patch(gear)
    
    tooth_depth = radius * 0.1
    for i in range(num_teeth):
        tooth_angle = angle + i * 2 * np.pi / num_teeth
        x1 = center_x + radius * np.cos(tooth_angle)
        y1 = center_y + radius * np.sin(tooth_angle)
        x2 = center_x + (radius + tooth_depth) * np.cos(tooth_angle)
        y2 = center_y + (radius + tooth_depth) * np.sin(tooth_angle)
        plt.plot([x1, x2], [y1, y2], color=color)

def get_point_A_position(phi):
    xA = (R - r) * np.cos(phi)
    yA = (R - r) * np.sin(phi)
    return xA, yA

def get_point_A_velocity(phi, phi_dot):
    xA_dot = -(R - r) * phi_dot * np.sin(phi)
    yA_dot = (R - r) * phi_dot * np.cos(phi)
    return xA_dot, yA_dot

def get_point_A_acceleration(phi, phi_dot, phi_ddot):
    xA_ddot = -(R - r) * (phi_ddot * np.sin(phi) + phi_dot**2 * np.cos(phi))
    yA_ddot = (R - r) * (phi_ddot * np.cos(phi) - phi_dot**2 * np.sin(phi))
    return xA_ddot, yA_ddot

initial_state = [theta0, theta_dot0, phi0, phi_dot0]
solution = odeint(equations_of_motion, initial_state, t)

def update(frame):
    plt.clf()
    
    theta = solution[frame, 0]
    theta_dot = solution[frame, 1]
    phi = solution[frame, 2]
    phi_dot = solution[frame, 3]
    
    xA, yA = get_point_A_position(phi)
    vx, vy = get_point_A_velocity(phi, phi_dot)
    ax, ay = get_point_A_acceleration(phi, phi_dot, equations_of_motion(solution[frame], t[frame])[3])
    
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal')
    plt.grid(True)
    
    draw_gear(0, 0, R, theta, num_teeth=30, color='blue')
    draw_gear(xA, yA, r, -theta*R/r, num_teeth=10, color='red')
    
    plt.plot([0, xA], [0, yA], 'k-', linewidth=2)
    
    plt.plot([0, R*np.cos(theta)], [0, R*np.sin(theta)], 'b--', alpha=0.5)
    plt.plot([xA, xA + r*np.cos(-theta*R/r)], [yA, yA + r*np.sin(-theta*R/r)], 'r--', alpha=0.5)
    
    v_scale = 0.5
    if np.hypot(vx, vy) > 0.001:
        plt.arrow(xA, yA, vx*v_scale, vy*v_scale, 
                 head_width=0.05, head_length=0.1, fc='g', ec='g', label='Velocity')
    
    a_scale = 0.2
    if np.hypot(ax, ay) > 0.001:
        plt.arrow(xA, yA, ax*a_scale, ay*a_scale, 
                 head_width=0.05, head_length=0.1, fc='r', ec='r', label='Acceleration')
    
    plt.plot(0, 0, 'ko', markersize=10)
    plt.plot(xA, yA, 'ko', markersize=10)
    plt.text(-0.2, -0.2, 'O', fontsize=12)
    plt.text(xA-0.2, yA-0.2, 'A', fontsize=12)
    
    plt.legend()
    plt.title(f't = {t[frame]:.2f} s\nθ = {theta:.2f} rad, φ = {phi:.2f} rad')

fig = plt.figure(figsize=(10, 10))
anim = FuncAnimation(fig, update, frames=len(t), interval=20, repeat=True)
plt.show()