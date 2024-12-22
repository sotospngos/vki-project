from scipy import interpolate
import numpy as np
from scipy.optimize import minimize

# Νέα συνάρτηση για υπολογισμό Cq και Cp
def interp_cq_cp(pitch, TSR, cq_table, cp_table, pitch_initial_rad, TSR_initial):
    """
    Παρεμβολή για την εύρεση του Cq και Cp σε δεδομένη γωνία βήματος και TSR.

    Parameters:
    -----------
    pitch: float
        Γωνία βήματος (rad).
    TSR: float
        Tip-Speed Ratio (rad).
    cq_table: array_like
        Πίνακας δεδομένων συντελεστών ροπής (Cq).
    cp_table: array_like
        Πίνακας δεδομένων συντελεστών ισχύος (Cp).
    pitch_initial_rad: array_like
        Γωνίες βήματος σε ακτίνια (x-axis του πίνακα).
    TSR_initial: array_like
        Tip-Speed Ratios (y-axis του πίνακα).

    Returns:
    --------
    cq: float
        Συντελεστής ροπής (Cq) στην αντίστοιχη γωνία βήματος και TSR.
    cp: float
        Συντελεστής ισχύος (Cp) στην αντίστοιχη γωνία βήματος και TSR.
    """
    interpolant_cq = interpolate.RectBivariateSpline(
        pitch_initial_rad, TSR_initial, cq_table.T
    )
    interpolant_cp = interpolate.RectBivariateSpline(
        pitch_initial_rad, TSR_initial, cp_table.T
    )
    cq = interpolant_cq(pitch, TSR)[0, 0]  # Διορθωμένη εξαγωγή τιμής
    cp = interpolant_cp(pitch, TSR)[0, 0]  # Διορθωμένη εξαγωγή τιμής
    return cq, cp


# Ενημερωμένη συνάρτηση προσομοίωσης
def simulate_wind_turbine_with_params(
    t_array, ws_array, wd_array, rho, R, GenEff, Ng, J, bld_pitch, gen_torque,
    cq_table, cp_table, pitch_initial_rad, TSR_initial
):
    dt = t_array[1] - t_array[0]
    rot_speed = np.zeros_like(t_array)
    gen_speed = np.zeros_like(t_array)
    gen_power = np.zeros_like(t_array)
    aero_torque = np.zeros_like(t_array)

    rot_speed[0] = 0.1  # Αρχική ταχύτητα ρότορα

    for i, t in enumerate(t_array):
        if i == 0:
            continue

        ws = ws_array[i]
        tsr = rot_speed[i-1] * R / ws
        if tsr == 0:
            tsr = 1e-6

        # Υπολογισμός Cq και Cp
        cq, cp = interp_cq_cp(
            bld_pitch, tsr, cq_table, cp_table, pitch_initial_rad, TSR_initial
        )

        aero_torque[i] = 0.5 * rho * (np.pi * R**3) * (cp / tsr) * ws**2
        rot_speed[i] = rot_speed[i-1] + (dt / J) * (aero_torque[i] * GenEff / 100 - Ng * gen_torque)
        gen_speed[i] = rot_speed[i] * Ng
        gen_power[i] = gen_speed[i] * gen_torque * GenEff / 100

    return gen_power

# Στόχος για βελτιστοποίηση
def objective_function(
    params, t_array, ws_array, wd_array, rho, R, GenEff, Ng, J,
    cq_table, cp_table, pitch_initial_rad, TSR_initial
):
    bld_pitch, gen_torque = params
    power_output = simulate_wind_turbine_with_params(
        t_array, ws_array, wd_array, rho, R, GenEff, Ng, J, bld_pitch, gen_torque,
        cq_table, cp_table, pitch_initial_rad, TSR_initial
    )
    average_power = np.mean(power_output)
    return -average_power

# Κύριο πρόγραμμα
if __name__ == "__main__":
    # Δεδομένα εισαγωγής
    t_array = np.linspace(0, 600, 601)  # Χρονική διάταξη (s)
    ws_array = np.ones_like(t_array) * 12  # Σταθερός άνεμος 12 m/s
    wd_array = np.ones_like(t_array) * 270  # Σταθερή διεύθυνση ανέμου

    # Σταθερές
    rho = 1.225  # Πυκνότητα αέρα
    R = 63  # Ακτίνα ρότορα
    GenEff = 94  # Απόδοση γεννήτριας
    Ng = 97  # Αναλογία ταχυτήτων
    J = 4.1e6  # Ροπή αδράνειας

    # Πίνακες δεδομένων
    cq_table = np.random.random((10, 10))  # Υποθετικός πίνακας Cq
    cp_table = np.random.random((10, 10))  # Υποθετικός πίνακας Cp
    pitch_initial_rad = np.linspace(0, 30, 10) * np.pi / 180  # Γωνίες βήματος (rad)
    TSR_initial = np.linspace(4, 12, 10)  # Tip-Speed Ratios (rad)

    # Αρχικές τιμές παραμέτρων
    initial_params = [2, 1e4]  # [bld_pitch, gen_torque]

    # Όρια παραμέτρων
    bounds = [
        (0, 30),       # Όριο για τη γωνία βήματος (deg)
        (5000, 2e4)    # Όριο για τη ροπή γεννήτριας (Nm)
    ]

    # Εκτέλεση βελτιστοποίησης
    result = minimize(
        objective_function,
        initial_params,
        args=(t_array, ws_array, wd_array, rho, R, GenEff, Ng, J,
              cq_table, cp_table, pitch_initial_rad, TSR_initial),
        bounds=bounds,
        method="L-BFGS-B"
    )

    # Αποτελέσματα
    optimal_params = result.x
    optimal_average_power = -result.fun  # Αφαίρεση του αρνητικού για την πραγματική μέγιστη ισχύ

    print(f"Βέλτιστες Παράμετροι: Γωνία Βήματος = {optimal_params[0]:.2f} rad, Ροπή Γεννήτριας = {optimal_params[1]:.2f} Nm")
    print(f"Μέγιστη Μέση Παραγόμενη Ισχύς: {optimal_average_power:.2f} W")
