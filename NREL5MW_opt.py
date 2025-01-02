import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from cp_NREL5MW import TSR_initial, pitch_initial_deg, performance_table

# Unit conversions
deg2rad = np.pi / 180.0
rpm2RadSec = 2 * np.pi / 60.0

# Convert pitch_initial_deg to radians
pitch_initial_rad = pitch_initial_deg * deg2rad

# Turbine parameters (taken from the YAML file)
R = 63.0  # Rotor radius (m)
J = 38677040.613  # Rotor inertia (kg*m^2)
GBRatio = 97  # Gearbox ratio
GenEff = 95  # Generator efficiency (%)

# 2D interpolation function
def interp_surface(performance_table, pitch_initial_rad, TSR_initial, pitch, TSR):
    """
    2D interpolation to find a value on the rotor performance surface.
    """
    interp_fun = RectBivariateSpline(pitch_initial_rad, TSR_initial, performance_table.T)
    return float(np.squeeze(interp_fun(pitch, TSR)))

# Turbine simulation function
def turbine_sim(t_array, ws_array, R, rho, J, GBRatio, GenEff, wd_array, 
                performance_table, pitch_initial_rad, TSR_initial, 
                gen_torque, blade_pitch, rotor_rpm_init=10, yaw_init=0):
    """
    Simulates the turbine to calculate generator power given torque and blade pitch.
    Parameters and configuration values are based on the NREL 5MW turbine YAML file.
    """
    dt = t_array[1] - t_array[0]
    bld_pitch = np.ones_like(t_array) * blade_pitch * deg2rad
    rot_speed = np.ones_like(t_array) * rotor_rpm_init * rpm2RadSec  # rad/s
    gen_speed = np.ones_like(t_array) * rotor_rpm_init * GBRatio * rpm2RadSec  # rad/s
    aero_torque = np.ones_like(t_array) * 1000.0
    gen_power = np.ones_like(t_array) * 0.0
    nac_yaw = np.ones_like(t_array) * yaw_init
    nac_yawerr = np.ones_like(t_array) * 0.0

    for i, t in enumerate(t_array):
        if i == 0:
            continue  # Skip the first run
        ws = ws_array[i]
        wd = wd_array[i]

        # Calculate TSR (Tip-Speed Ratio)
        tsr = rot_speed[i-1] * R / ws

        # Interpolate Cp from the performance table
        cp = interp_surface(performance_table, pitch_initial_rad, TSR_initial, bld_pitch[i-1], tsr)

        # Update turbine state
        aero_torque[i] = 0.5 * rho * (np.pi * R**3) * (cp / tsr) * ws**2
        rot_speed[i] = rot_speed[i-1] + (dt / J) * (aero_torque[i] * GenEff / 100 - GBRatio * gen_torque)
        gen_speed[i] = rot_speed[i] * GBRatio
        nac_yawerr[i] = wd - nac_yaw[i-1]

        # Calculate power
        gen_power[i] = gen_speed[i] * gen_torque * GenEff / 100

        # Update nacelle position
        nac_yaw[i] = nac_yaw[i-1] + nac_yawerr[i] * dt

    return gen_power

# Objective function for optimization
def objective_function(params, t_array, ws_array, R, rho, J, GBRatio, GenEff, wd_array, 
                       performance_table, pitch_initial_rad, TSR_initial):
    """
    Objective function to maximize average generator power.
    """
    gen_torque, blade_pitch = params
    gen_power = turbine_sim(
        t_array, ws_array, R, rho, J, GBRatio, GenEff, wd_array, 
        performance_table, pitch_initial_rad, TSR_initial, 
        gen_torque, blade_pitch
    )
    avg_power = -np.mean(gen_power)  # Negative for minimization
    return avg_power

# Simulation data
t_array = np.linspace(0, 10, 100)  # Time array (s)
ws_array = np.linspace(8, 12, 100)  # Wind speed array (m/s)
wd_array = np.zeros_like(ws_array)  # Wind direction array (degrees)
rho = 1.225  # Air density (kg/m^3)

# Initial parameter guesses (generator torque and blade pitch)
initial_guess = [1000, 0]  # gen_torque (Nm), blade_pitch (deg)

# Bounds for parameters
bounds = [(500, 40000), (-5, 20)]  # gen_torque (Nm), blade_pitch (deg)

# Optimization
result = minimize(
    objective_function, 
    initial_guess, 
    args=(t_array, ws_array, R, rho, J, GBRatio, GenEff, wd_array, 
          performance_table, pitch_initial_rad, TSR_initial),
    bounds=bounds,
    method='L-BFGS-B'
)

# Extract optimization results
optimized_gen_torque, optimized_blade_pitch = result.x
optimized_avg_power = -result.fun  # Convert to positive value for maximized power

# Display results
print(f"Optimized Generator Torque: {optimized_gen_torque} Nm")
print(f"Optimized Blade Pitch: {optimized_blade_pitch} deg")
print(f"Maximized Average Power: {optimized_avg_power} W")
