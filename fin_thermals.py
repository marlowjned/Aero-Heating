# fin_thermals.py
# Rocket fin transient thermal estimator
# Aerothermal heating via oblique shock + Eckert/Stanton method (main.py)
# Transient 1D forward-Euler diffusion along fin chord (Nosecone Transient notebook)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

FILENAME = 'e3ref.csv'

# ---------------------------------------------------------------------------
# GAS CONSTANTS
# ---------------------------------------------------------------------------
g  = 1.4    # specific heat ratio (air)
R  = 287    # J/(kg·K)
cp = 1005   # J/(kg·K)
Pr = 0.71   # Prandtl number

# ---------------------------------------------------------------------------
# FIN AEROTHERMAL PARAMETERS (oblique shock geometry)
# ---------------------------------------------------------------------------
delta = 10 * np.pi / 180   # fin half-angle / deflection angle [rad]
theta = 10 * np.pi / 180   # flow incidence angle [rad]

# ---------------------------------------------------------------------------
# FIN MATERIAL PROPERTIES  (carbon fiber / CFRP)
# ---------------------------------------------------------------------------
rho_fin       = 1600   # kg/m³
cp_fin        = 800    # J/(kg·K)
k_fin         = 5      # W/(m·K)  (through-plane; ~50+ along fiber direction)
T_init_fin    = 300    # K  (ambient initial temperature)

# ---------------------------------------------------------------------------
# FIN GEOMETRY
# ---------------------------------------------------------------------------
fin_chord     = 0.15   # m  (along-flow dimension; model spatial domain)
fin_span      = 0.10   # m
fin_thickness = 0.003  # m

# ---------------------------------------------------------------------------
# RADIATION
# ---------------------------------------------------------------------------
boltz      = 5.67e-8   # W/(m²·K⁴)
emissivity = 0.90      # carbon fiber (bare CFRP is highly emissive)

# ---------------------------------------------------------------------------
# AEROTHERMAL HELPERS
# ---------------------------------------------------------------------------

def mu(T):
    """Dynamic viscosity of air from Sutherland's law"""
    mu0 = 1.716e-5   # Pa*s at T0
    T0  = 273.15     # K
    S   = 110.4      # K  (Sutherland constant)
    return mu0 * (T / T0) ** (3 / 2) * (T0 + S) / (T + S)


def _oblique_shock_analytical(M_inf, P_inf, T_inf, rho_inf):
    """Closed-form cubic root approach (for reference:
    arccos leaves [-1,1] for a lot of flight)."""
    b = -(M_inf**2 + 2) / M_inf**2 - g * np.sin(theta)**2
    c = ((2 * M_inf**2 + 1) / M_inf**4
         + ((g + 1)**2 / 4 + (g - 1) / M_inf**2) * np.sin(theta)**2)
    phi = np.arccos(
        (9 / 2 * b * c - b**3 + 27 / 2 * np.cos(theta / M_inf**2))
        / (b**2 - 3 * c) ** (3 / 2)
    )
    sin2B = b / 3 + 2 / 3 * (b**2 - 3 * c) ** 0.5 * np.cos((phi + 4 * np.pi) / 3)
    B = np.arcsin(sin2B ** 0.5)
    MN1 = M_inf * np.sin(B)
    MN2 = np.sqrt((MN1**2 + 2 / (g - 1)) / (2 * g / (g - 1) * MN1**2))
    p_ratio   = (2 * g * MN1**2 - (g - 1)) / (g + 1)
    T_ratio   = p_ratio * ((g - 1) * MN1**2 + 2) / ((g + 1) * MN1**2)
    rho_ratio = (g + 1) * MN1**2 / ((g - 1) * MN1**2 + 2)
    Me   = MN2 / np.sin(B - delta)
    pe   = P_inf * p_ratio
    Te   = T_inf * T_ratio
    rhoe = rho_inf * rho_ratio
    ae   = np.sqrt(g * R * Te)
    Ve   = Me * ae
    he   = cp * Te
    return Me, pe, Te, rhoe, Ve, he


def oblique_shock(M_inf, P_inf, T_inf, rho_inf):
    """Compute post-shock edge conditions using the theta-beta-Mach relation."""
    def tbm(B):
        return (np.tan(theta) * (M_inf**2 * (g + np.cos(2 * B)) + 2)
                - 2 / np.tan(B) * (M_inf**2 * np.sin(B)**2 - 1))

    # Endpoints (that are computable)
    B_min = np.arcsin(1.0 / M_inf) + 1e-6
    B_max = np.pi / 2.0 - 1e-6

    # tbm goes +→0→-→0→+ across [B_min, B_max], so both endpoints are positive
    # and brentq won't find a root unless we bracket the first + → - crossing.
    # Scan for the weak-shock bracket (first sign change from + to -).
    n = 100
    B_scan = np.linspace(B_min, B_max, n)
    f_scan = np.array([tbm(b) for b in B_scan])
    bracket = None
    for i in range(n - 1):
        if f_scan[i] > 0 and f_scan[i + 1] < 0:
            bracket = (B_scan[i], B_scan[i + 1])
            break
    if bracket is None:
        return None   # detached shock — no weak-shock solution exists

    try:
        B = brentq(tbm, bracket[0], bracket[1], full_output=False)
    except ValueError:
        return None

    MN1 = M_inf * np.sin(B)
    MN2 = np.sqrt((MN1**2 + 2 / (g - 1)) / (2 * g / (g - 1) * MN1**2))

    p_ratio   = (2 * g * MN1**2 - (g - 1)) / (g + 1)
    T_ratio   = p_ratio * ((g - 1) * MN1**2 + 2) / ((g + 1) * MN1**2)
    rho_ratio = (g + 1) * MN1**2 / ((g - 1) * MN1**2 + 2)

    Me   = MN2 / np.sin(B - delta)
    pe   = P_inf * p_ratio
    Te   = T_inf * T_ratio
    rhoe = rho_inf * rho_ratio
    ae   = np.sqrt(g * R * Te)
    Ve   = Me * ae
    he   = cp * Te

    return Me, pe, Te, rhoe, Ve, he


def compute_qw(M_inf, P_inf, T_inf, rho_inf, x, Tw):
    """
    Heat flux at fin surface [W/m^2] at chord-wise position x [m] from
    the leading edge, given freestream conditions and wall temperature Tw [K].

    Uses oblique shock for edge conditions, then Eckert reference-enthalpy
    method with turbulent flat-plate Stanton number correlation.
    """
    x = max(x, 1e-6)   # avoid singularity at leading edge

    shock = oblique_shock(M_inf, P_inf, T_inf, rho_inf)
    if shock is None:
        return 0.0   # skip heating

    _, pe, _, _, Ve, he = shock

    # Recovery factor and adiabatic wall conditions
    r   = Pr ** (1 / 3)
    haw = he + r * Ve**2 / 2
    hw  = cp * Tw

    # Eckert reference enthalpy -> reference temperature
    h_ref = 0.5 * (hw + he) + 0.22 * (haw - he)
    T_ref = h_ref / cp

    rho_ref = pe / (R * T_ref)

    mu_ref = mu(T_ref)
    k_ref  = mu_ref * cp / Pr
    Pr_ref = mu_ref * cp / k_ref   # ≈ Pr (included for completeness)

    Rex_ref = rho_ref * Ve * x / mu_ref

    # Turbulent flat-plate Stanton number
    St = 0.0296 * Rex_ref ** (-0.2) * Pr_ref ** (-2 / 3)

    qw = St * rho_ref * Ve * (haw - hw)   # [W/m^2]
    return qw


# ---------------------------------------------------------------------------
# TRANSIENT 1D THERMAL MODEL  (forward Euler, fin chord direction)
# ---------------------------------------------------------------------------

class ThermalModel1D:
    """ 1D transient thermal model: 
    heating into nodes, conduction through fins, and radiative cooling"""

    def __init__(self, length, num_points,
                 thickness=fin_thickness,
                 span=fin_span,
                 rho=rho_fin,
                 cp_mat=cp_fin,
                 k_mat=k_fin,
                 T_init=T_init_fin,
                 eps=emissivity):
        self.length     = length
        self.num_points = num_points
        self.dx         = length / (num_points - 1)
        self.x          = np.linspace(0, length, num_points)
        self.T          = np.ones(num_points) * T_init

        # Material
        self.rho    = rho
        self.cp_mat = cp_mat
        self.k_mat  = k_mat
        self.eps    = eps

        # Flat-plate geometry (constant along chord)
        self.Ac  = thickness * span           # conduction cross-section [m²]
        self.dAs = 2.0 * self.dx * span       # wetted surface per node (both sides) [m²]
        self.dV  = thickness * self.dx * span  # volume per node [m³]

    def step(self, dt, M_inf, P_inf, T_inf, rho_inf):
        """ Euler's method step """
        Tm = self.T.copy()
        N  = len(Tm)

        for m in range(N):
            x  = self.x[m]
            qw = compute_qw(M_inf, P_inf, T_inf, rho_inf, x, Tm[m])

            cond_left  = (self.k_mat / self.dx * self.Ac * (Tm[m - 1] - Tm[m])
                          if m > 0 else 0.0)
            cond_right = (self.k_mat / self.dx * self.Ac * (Tm[m + 1] - Tm[m])
                          if m < N - 1 else 0.0)

            q_rad = self.eps * boltz * (Tm[m]**4 - T_inf**4)   # W/m²

            self.T[m] = (Tm[m]
                         + dt / (self.rho * self.cp_mat * self.dV)
                         * (cond_left + cond_right + (qw - q_rad) * self.dAs))

    def solve(self, filename):
        """
        Load openrocket flight data, and run a transient sim
        Expected columns:
            '# Time (s)', 'Altitude (m)', 'Total velocity (m/s)',
            'Mach number (​)', 'Air temperature (°C)', 'Air pressure (mbar)'
        """
        data = pd.read_csv(filename)

        history    = [self.T.copy()]
        stop_index = len(data) - 1

        for t in range(len(data) - 1):
            dt = data['# Time (s)'][t + 1] - data['# Time (s)'][t]
            if dt <= 0:
                continue

            row   = data.loc[t]
            M     = row['Mach number (​)']
            P     = row['Air pressure (mbar)'] * 100          # Pa
            T_air = row['Air temperature (°C)'] + 273.15      # K
            rho   = P / (R * T_air)

            # Only heat during supersonic flight
            if M > 1.0:
                self.step(dt, M, P, T_air, rho)

            history.append(self.T.copy())

            # Stop at apogee
            if data['Altitude (m)'][t + 1] < data['Altitude (m)'][t]:
                stop_index = t + 1
                break

        times   = np.array(data['# Time (s)'][:stop_index + 1])
        history = np.array(history[:stop_index + 1])
        return history, times


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    model = ThermalModel1D(length=fin_chord, num_points=50)
    history, times = model.solve(FILENAME)

    # Plot temperature vs time at several chord-wise nodes
    plot_nodes = [0, 10, 20, 30, 49]
    for i in plot_nodes:
        x_label = model.x[i] * 100  # cm
        plt.plot(times, history[:, i], label=f'x = {x_label:.1f} cm')

    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [K]')
    plt.title('Fin surface temperature vs time (chord-wise nodes)')
    plt.legend()
    plt.tight_layout()
    plt.show()
