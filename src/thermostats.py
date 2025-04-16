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