# main.py
# Rocket airframe thermal estimator

import numpy as np
from scipy.optimize import brentq

# FIN THERMALS

# --- theta beta mach ---
# USER INPUT PARAMS
x = 0.1 # distance from tip

# orientation
delta = 10*np.pi/180
theta = 10*np.pi/180

# Air properties
g = 1.4
R = 287
cp = 1005
Pr = 0.71

# Freestream conditions
M_INF = 6
P_INF = 101325
T_INF = 288.15
RHO_INF = 1.225

# SHOCK ANGLE CALCULATION
# Eckhart's solution
def eckhart_func() -> float:
    b = -(M_INF**2 + 2)/(M_INF**2) - g*np.sin(theta)**2
    c = (2*M_INF**2 + 1)/M_INF**4 + ((g + 1)**2 /4 + (g - 1)/M_INF**2)*np.sin(theta)**2
    phi = np.arccos(9/2*b*c - b**3 + 27/2*np.cos(theta/M_INF**2))/(b**2 - 3*c)**(3/2)

    sin2B = b/3 + 2/3*((b**2 -3*c)**0.5)*np.cos((phi + 4*np.pi)/3)
    B = np.arcsin(sin2B**0.5)
    return B

# Alternative: brentq solution
def shock_angle_func():
    def func(B):
        return np.tan(theta) - 2*np.tan(B)*((M_INF**2*np.sin(B)**2 - 1)/(M_INF**2*(g + np.cos(2*B)) + 2)) 
    B = brentq(func, theta, np.pi/2)
    return B

B = eckhart_func()

# Normal component @ this calculation
MN1 = M_INF*np.sin(B)
MN2 = np.sqrt((MN1**2 + 2/(g-1)))/(2*g/(g-1)*MN1**2)

p_ratio = (2*g*MN1**2 - (g-1))/(g+1)
T_ratio = (2*g*MN1**2 - (g-1))/(g+1)*((g-1)*MN1**2 + 2)/((g+1)*MN1**2)
rho_ratio = ((g+1)*MN1**2)/((g-1)*MN1**2 + 2)

# Boundary layer conditions
Me = MN2/np.sin(B - delta)
pe = P_INF*p_ratio
Te = T_INF*T_ratio
rhoe = RHO_INF*rho_ratio
ae = np.sqrt(g*R*Te)
Ve = Me*ae
he = cp*Te

r = Pr**(1/3)
haw = r*Ve**2/2
hw = 0 ### WOULD BE IN LOOP
Taw = haw/cp
Tw = hw/cp

h_ref = 0.5*(hw - he) + 0.22*(haw - he)
T_ref = h_ref/cp
rho_ref = pe/(R*T_ref)

def mu(T):
    # Sutherland's law
    mu_ref = 1.716e-5
    T_ref = 273.15
    S = 110.4
    return mu_ref*(T/T_ref)**(3/2)*(T_ref + S)/(T + S)

k_ref = mu(T_ref)*cp/Pr
Pr_ref = mu(T_ref)*cp/k_ref
Rex_ref = rho_ref*Ve*x/mu(T_ref) 

St = 0.0296*Rex_ref**(-0.2)*(Pr_ref)**(-2/3) 
#if turbulent else 0.664*Rex_ref**(-0.5)*(Pr_ref)**(-2/3)

qw = St*rho_ref*Ve*(Taw - Tw)

# -kwall * dT/dz(@ z=0) = qw

# --- prandtl-meyer to fin face tube ---

def prandtl_meyer(M):
    nu = np.sqrt((g+1)/(g-1))*np.arctan(np.sqrt((g-1)/(g+1)*(M**2 - 1))) - np.arctan(np.sqrt(M**2 - 1))
    return nu

def invert_prandtl_meyer(nu_target, tol=1e-8):
    M_low = 0.8   # absolute min mach number
    M_high = 5.0  # shouldn't use this tool for hypersonics
    
    while (M_high - M_low) > tol:
        M_mid = 0.5 * (M_low + M_high)
        if prandtl_meyer(M_mid) < nu_target:
            M_low = M_mid
        else:
            M_high = M_mid
    
    return 0.5 * (M_low + M_high)

p_ratio_face = (1 + (g-1)/2*M_INF**2)**(g/(g-1))/((1 + (g-1)/2*MN1**2)**(g/(g-1)))
T_ratio_face = (1 + (g-1)/2*M_INF**2)/((1 + (g-1)/2*MN2**2))

p2 = P_INF*p_ratio_face
T2 = T_INF*T_ratio_face
a2 = np.sqrt(g*R*T2)
V2 = MN2*a2

# find v(M2) = v(M1) + delta


# prandtl-meyer to bottom taper



# --- Make 1D thermal model ---
class ThermalModel1D:
    def __init__(self, length, num_points):
        self.length = length
        self.num_points = num_points
        self.k = 0.1 # TODO: Allow multiple materials
        self.z = np.linspace(0, length, num_points)
        self.T = np.zeros(num_points)  # Temperature distribution along the length

    def solve(self, filename: str):

        pass

    def rk4step(self, qw: float):
        derivative = -qw / self.k  # dT/dz = -qw / k
        
        pass


# --- Solve 1D thermal model ---
model = ThermalModel1D(length=1.0, num_points=100)
model.solve("p1ref.csv")
