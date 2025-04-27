import numpy as np

def langevin_thermostat(velocities, dt, temperature, friction_coef, mass, kb):
    """Apply Langevin thermostat."""
    # Random acceleration (m/s2)
    random_vel = np.sqrt(2 * friction_coef * kb * temperature * dt) * np.random.normal(0, 1, velocities.shape) / mass
    # Dissipative acceleration (m/s2)
    dissipative_vel = - friction_coef / mass * velocities * dt
    # Update velocities
    return velocities + dissipative_vel + random_vel

def berendsen_thermostat(velocities, dt, target_temperature, tau, n_particles, mass, kb):
    """Apply Berendsen thermostat to rescale velocities."""
    # tau is the coupling time constant in seconds
    dimensions = 3
    # Compute kinetic energy in Joules
    kinetic_energy = 0.5 * mass * np.sum(velocities ** 2)
    # Calculate current temperature in Kelvin
    # DOF = 3N
    dof = dimensions * n_particles 
    current_temperature = (2 * kinetic_energy) / (dof * kb)
    # Calculate scaling factor lambda
    lambda_factor = np.sqrt(1 + (dt / tau) * ((target_temperature / current_temperature) - 1))
    return velocities * lambda_factor

def nose_hoover_thermostat(velocities, dt, temperature, kb, mass, Q, zeta):
    """
    Nosé-Hoover thermostat implementation.
    
    Args:
        velocities: Array of particle velocities (m/s)
        dt: Time step (s)
        temperature: Target temperature (K)
        kb: Boltzmann constant (J/K)
        mass: Particle mass (kg)
        Q: Thermostat coupling constant (controls response speed)
        zeta: Thermostat friction coefficient (updated during simulation)
    
    Returns:
        Updated velocities and new zeta value
    """
    n_particles = velocities.shape[0]
    dimensions = 3
    # Calculate kinetic energy
    kinetic_energy = 0.5 * mass * np.sum(velocities**2)
    # Update zeta (friction coefficient)
    current_temp = kinetic_energy / (1.5 * n_particles * kb)
    zeta_dot = (1/Q) * (kinetic_energy - 1.5 * n_particles * kb * temperature)
    new_zeta = zeta + zeta_dot * dt
    # Apply velocity scaling
    scaling_factor = np.exp(-new_zeta * dt)
    updated_velocities = velocities * scaling_factor
    return updated_velocities, new_zeta

def parrinello_rahman_barostat(
    positions, velocities, forces, dt, pressure, W,
    box_size, mass, inst_pressure, eta
):
    """
    Scalar Parrinello-Rahman barostat (isotropic, cubic box).
    
    Args:
        positions: Nx3 array (meters)
        velocities: Nx3 array (m/s)
        forces: Nx3 array (N)
        dt: timestep (s)
        pressure: target pressure (Pa)
        W: barostat mass/coupling (kg m^2)
        box_size: box length (meters)
        mass: particle mass (kg)
        inst_pressure: instantaneous system pressure (Pa)
        eta: current barostat velocity (1/s)
        
    Returns:
        updated_positions, updated_velocities, new_box_size, new_eta
    """
    # 1. Barostat velocity half-step update
    volume = box_size ** 3
    eta += 0.5 * dt * (volume / W) * (inst_pressure - pressure)
    
    # 2. Scale box and positions (full step: L = L * exp(eta*dt))
    scale = np.exp(dt * eta)
    new_box_size = box_size * scale
    updated_positions = positions * scale
    updated_velocities = velocities * scale  # velocities scale as positions
    
    return updated_positions, updated_velocities, new_box_size, eta
    
# def parrinello_rahman_barostat(positions, velocities, forces, dt, pressure, W, box_size, mass, virial_sum=None):
#     """
#     Optimized Parrinello-Rahman barostat for constant pressure simulations.
#     """
#     # Precompute values used multiple times
#     volume = box_size ** 3
#     inv_volume = 1.0 / volume
#     inv_box_size = 1.0 / box_size
#     dt_squared = dt ** 2

#     # Calculate kinetic contribution to pressure
#     # Use in-place operations where possible
#     kin_sum = np.sum(velocities * velocities)  # Faster than velocities**2
#     kinetic_pressure = mass * kin_sum * inv_volume / 3.0

#     # Calculate virial contribution to pressure
#     if virial_sum is None:
#         # Vectorized dot product
#         virial_sum = np.sum(positions * forces)
#     virial_pressure = -virial_sum * inv_volume / 3.0

#     # Total instantaneous pressure
#     inst_pressure = kinetic_pressure + virial_pressure

#     # Pressure difference and box acceleration
#     pressure_diff = inst_pressure - pressure
#     box_accel = volume * pressure_diff / W

#     # Calculate scaling factor (avoiding division when possible)
#     scaling_factor = 1.0 - (dt_squared * box_accel * inv_box_size) / 3.0

#     # Limit scaling (np.clip is more efficient than min/max)
#     #scaling_factor = np.clip(scaling_factor, 0.95, 1.05)

#     # Update box size (cbrt is computationally expensive, use power of 1/3)
#     new_box_size = box_size * scaling_factor**(1.0/3.0)
#     #new_box_size = max(new_box_size, box_size * 0.5)

#     # Calculate scale ratio once
#     scale_ratio = new_box_size * inv_box_size

#     # Scale positions and velocities (in-place multiplication is faster)
#     updated_positions = positions * scale_ratio
#     updated_velocities = velocities * scale_ratio

#     return updated_positions, updated_velocities, new_box_size, scale_ratio
