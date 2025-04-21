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

def parrinello_rahman_barostat(positions, velocities, forces, dt, pressure, W, box_size, mass):
    """
    Parrinello-Rahman barostat for constant pressure simulations.
    
    Args:
        positions: Array of particle positions (m)
        velocities: Array of particle velocities (m/s)
        forces: Array of particle forces (N)
        dt: Time step (s)
        pressure: Target pressure (Pa)
        W: Barostat coupling constant (controls response speed)
        box_size: Current box size (m)
        mass: Particle mass (kg)
    
    Returns:
        Updated positions, velocities, box size, and scaling factor
    """
    n_particles = positions.shape[0]
    dimensions = 3
    
    # Calculate instantaneous pressure (simplified)
    kinetic_pressure = (1/(3*box_size**3)) * np.sum(mass * velocities**2)
    virial_pressure = (1/(3*box_size**3)) * np.sum(positions * forces)
    inst_pressure = kinetic_pressure + virial_pressure
    
    # Calculate barostat scaling
    epsilon_dot = (1/W) * (inst_pressure - pressure)
    epsilon = epsilon_dot * dt
    scaling_factor = np.exp(epsilon)
    
    # Update positions and box size
    new_box_size = box_size * scaling_factor
    updated_positions = positions * scaling_factor
    updated_velocities = velocities * np.sqrt(scaling_factor)
    
    return updated_positions, updated_velocities, new_box_size, scaling_factor