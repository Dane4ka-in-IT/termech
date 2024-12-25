import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

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

initial_state = [theta0, theta_dot0, phi0, phi_dot0]
solution = odeint(equations_of_motion, initial_state, t)

def calculate_reactions(state, t):
    theta, theta_dot, phi, phi_dot = state
    _, theta_ddot, _, phi_ddot = equations_of_motion(state, t)
    
    NAx = m2*(R - r)*(phi_dot*np.cos(phi) - phi_ddot*np.sin(phi))
    NAy = m2*(-(R - r)*(phi_dot*np.sin(phi) + phi_ddot*np.cos(phi)) + g)
    
    return NAx, NAy

NAx = np.zeros_like(t)
NAy = np.zeros_like(t)

for i in range(len(t)):
    NAx[i], NAy[i] = calculate_reactions(solution[i], t[i])

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(t, NAx, 'b-', label='Horizontal Reaction Force')
plt.xlabel('Time (s)')
plt.ylabel('Reaction Force in X-direction (N)')
plt.title('Horizontal Reaction Force at Point A')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, NAy, 'r-', label='Vertical Reaction Force')
plt.xlabel('Time (s)')
plt.ylabel('Reaction Force in Y-direction (N)')
plt.title('Vertical Reaction Force at Point A')
plt.grid(True)
plt.legend()

plt.suptitle('Reaction Forces at Joint A Between Crank and Gear 2', fontsize=12, y=1.02)
plt.tight_layout()
plt.show()