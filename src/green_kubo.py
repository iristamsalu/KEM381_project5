import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import os

def unnormalized_autocorrelation(x):
    """
    Calculates the unnormalized autocorrelation function C(t) = <x(0)x(t)>.
    """
    n = len(x)
    # Calculate correlation using FFT for efficiency if n is large, direct otherwise
    # Direct method:
    result = np.correlate(x, x, mode='full')
    # Select the second half corresponding to lags 0 to n-1
    sacf = result[n - 1:] 
    
    # Normalize for the decreasing number of overlapping points at larger lags
    # This provides a statistically better estimate for longer times
    overlap_norm = np.arange(n, 0, -1) 
    sacf = sacf / overlap_norm
    
    return sacf # Units: units(x)^2, e.g., Pa^2

# --- Configuration & Constants ---
DATA_FILE = 'output/pressure.dat' 
VOLUME =5.240e-26         # Volume in m^3
KB = 1.380649e-23          # Boltzmann constant in J/K
TEMPERATURE = 107.7        # Temperature in K

# Time limit for integration (in ps) - Choose based on SACF decay
# Look at the SACF plot and choose a time after which it's mostly noise around zero.
# Start with a value like 10-20 ps and adjust.
INTEGRATION_TIME_PS = 100 

# --- Load Data ---
if not os.path.exists(DATA_FILE):
    print(f"Error: Data file '{DATA_FILE}' not found.")
    exit()
    
try:
    # Skip header row, usecols to select specific columns if needed
    data = np.loadtxt(DATA_FILE, skiprows=1) 
    if data.shape[1] < 5:
         raise ValueError("Data file should have at least 5 columns: Step, Time, Pxy, Pxz, Pyz")
    time_s = data[:, 1]   # Time in seconds
    pxy = data[:, 2]      # Pxy in Pascals
    pxz = data[:, 3]      # Pxz in Pascals
    pyz = data[:, 4]      # Pyz in Pascals
except Exception as e:
    print(f"Error loading data from '{DATA_FILE}': {e}")
    exit()

# Check for sufficient data
if len(time_s) < 2:
    print("Error: Not enough data points to calculate time step or autocorrelation.")
    exit()

# --- Data Processing ---
# Calculate time step (ensure uniform spacing)
dt_s = time_s[1] - time_s[0]
if dt_s <= 0:
    print("Error: Time step calculation failed or non-positive dt.")
    exit()
# Check if time steps are reasonably constant (optional but good)
time_diffs = np.diff(time_s)
if not np.allclose(time_diffs, dt_s, rtol=1e-3):
     print(f"Warning: Time steps in {DATA_FILE} may not be uniform.")
     # dt_s = np.mean(time_diffs) # Use mean dt if non-uniform

# Subtract mean from pressure components (calculate fluctuations P')
pxy_prime = pxy - np.mean(pxy)
pxz_prime = pxz - np.mean(pxz)
pyz_prime = pyz - np.mean(pyz)

# Calculate UNNORMALIZED Stress Autocorrelation Functions (SACF) in Pa^2
print("Calculating SACF...")
sacf_pxy = unnormalized_autocorrelation(pxy_prime)
sacf_pxz = unnormalized_autocorrelation(pxz_prime)
sacf_pyz = unnormalized_autocorrelation(pyz_prime)

# Average the three components (Pa^2)
sacf_avg = (sacf_pxy + sacf_pxz + sacf_pyz) / 3.0

# --- Green-Kubo Integration ---
# Calculate the running integral of the average SACF (Units: Pa^2 * s)
print("Integrating SACF...")
# Use cumulative_trapezoid for the running integral
running_integral_Pa2_s = cumulative_trapezoid(sacf_avg, dx=dt_s, initial=0)

# Calculate the Green-Kubo prefactor (Units: m^3 / J = m^3 / (N*m) = m^2 / N)
# Alternatively: m^3 / (kg*m^2/s^2) = m*s^2 / kg
gk_prefactor = VOLUME / (KB * TEMPERATURE) # Units: m*s^2/kg

# Calculate the running viscosity (Units: (m*s^2/kg) * (Pa^2 * s) = (m*s^2/kg) * ((N/m^2)^2 * s) )
eta_Pa_s = gk_prefactor * running_integral_Pa2_s

# Convert to microPascal-seconds (µPa·s) for plotting/reporting
eta_microPa_s = eta_Pa_s * 1e6

# --- Extract Final Viscosity Estimate ---
# Find the index corresponding to the desired integration time limit
integration_time_s = INTEGRATION_TIME_PS * 1e-12 # Convert ps to s
integration_steps = int(integration_time_s / dt_s)

# Ensure the index is within the bounds of the calculated data
if integration_steps >= len(eta_microPa_s):
    print(f"Warning: Integration time ({INTEGRATION_TIME_PS} ps) exceeds data length.")
    integration_steps = len(eta_microPa_s) - 1 # Use the last available point

# Extract the viscosity value at the integration limit
final_eta_microPa_s = eta_microPa_s[integration_steps]

# Average over a plateau window for a more stable estimate
plateau_start_ps = max(5.0, INTEGRATION_TIME_PS * 0.5) # Example: Start halfway to the limit, but at least 5ps
plateau_end_ps = INTEGRATION_TIME_PS
plateau_start_idx = int(plateau_start_ps * 1e-12 / dt_s)
plateau_end_idx = integration_steps + 1 # Include the endpoint step
if plateau_start_idx < plateau_end_idx:
     plateau_eta_microPa_s = np.mean(eta_microPa_s[plateau_start_idx:plateau_end_idx])
     print(f"Viscosity (Plateau Avg {plateau_start_ps:.1f}-{plateau_end_ps:.1f} ps): {plateau_eta_microPa_s:.4f} μPa·s")
else:
     plateau_eta_microPa_s = final_eta_microPa_s # Fallback if window is invalid
     print(f"Viscosity (Value at {INTEGRATION_TIME_PS:.1f} ps): {final_eta_microPa_s:.4f} μPa·s")


# --- Visualization ---
print("Generating plots...")
plt.figure(figsize=(10, 8)) # Adjusted size slightly

# --- Plot SACF ---
plt.subplot(2, 1, 1)
# Calculate time axis for correlation plot in ps
t_corr_ps = np.arange(len(sacf_avg)) * dt_s * 1e12 

# Limit plot range for better visibility of decay (e.g., 0 to 2*INTEGRATION_TIME_PS)
plot_limit_ps = min(t_corr_ps[-1], INTEGRATION_TIME_PS * 2.5) 
plot_limit_idx = int(plot_limit_ps / (dt_s * 1e12))

plt.plot(t_corr_ps[:plot_limit_idx], sacf_pxy[:plot_limit_idx], label='$P_{xy}$ component', alpha=0.7)
plt.plot(t_corr_ps[:plot_limit_idx], sacf_pxz[:plot_limit_idx], label='$P_{xz}$ component', alpha=0.7)
plt.plot(t_corr_ps[:plot_limit_idx], sacf_pyz[:plot_limit_idx], label='$P_{yz}$ component', alpha=0.7)
plt.plot(t_corr_ps[:plot_limit_idx], sacf_avg[:plot_limit_idx], 'k--', label='Average SACF')

plt.xlabel('Time (ps)')
plt.ylabel('SACF (Pa$^2$)')
plt.title('Unnormalized Stress Autocorrelation Functions')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
max_val_short_time = np.max(sacf_avg[1:int(1e-12/dt_s)]) # Max value after 1ps
# Zoom y-axis if initial peak is very large
# plt.ylim(-max_val_short_time*0.5, max_val_short_time*1.5) # Example zoom

# --- Plot Viscosity Integral ---
plt.subplot(2, 1, 2)
# Use the same time axis and plot limit
plt.plot(t_corr_ps[:plot_limit_idx], eta_microPa_s[:plot_limit_idx], label='Running Integral')

# Add lines indicating the chosen integration time and plateau window
plt.axvline(INTEGRATION_TIME_PS, color='r', linestyle='--', label=f'Integration Limit ({INTEGRATION_TIME_PS} ps)')
if plateau_start_idx < plateau_end_idx:
    plt.axhline(plateau_eta_microPa_s, color='g', linestyle=':', label=f'Plateau Avg ({plateau_eta_microPa_s:.2f} μPa·s)')
    # Shade the plateau region
    plt.axvspan(plateau_start_ps, plateau_end_ps, color='g', alpha=0.1)
else:
    plt.axhline(final_eta_microPa_s, color='g', linestyle=':', label=f'Value at Limit ({final_eta_microPa_s:.2f} μPa·s)')


plt.xlabel('Integration Time Limit (ps)')
plt.ylabel('Viscosity (μPa·s)')
plt.title('Viscosity from Green-Kubo Integration')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('viscosity_calculation.png')
print("Plots saved to viscosity_calculation.png")
