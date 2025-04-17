# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: extra_compile_args = -O3 -ffast-math

import numpy as np
cimport numpy as np
import cython
from libc.math cimport exp, sqrt, floor, round, pow, fabs, fmod, ceil
from libc.stdio cimport FILE, fopen, fwrite, fclose
from libc.stdlib cimport rand, RAND_MAX 

# Type definitions
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t INT_t

# Constants
cdef int RDF_BINS = 100
cdef DTYPE_t RDF_CUTOFF_FACTOR = 0.5
cdef int EMPTY = -1
cdef int COORD_DTYPE_SIZE = sizeof(DTYPE_t)
cdef int INT_DTYPE_SIZE = sizeof(int)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t compute_lj_force(DTYPE_t r, DTYPE_t sigma, DTYPE_t epsilon, DTYPE_t rcutoff) nogil:
    if r >= rcutoff or r < 1e-12:
        return 0.0
    cdef DTYPE_t inv_r = sigma / r
    cdef DTYPE_t inv_r6 = pow(inv_r, 6)
    cdef DTYPE_t inv_r12 = inv_r6 * inv_r6
    return 24 * epsilon * (2 * inv_r12 - inv_r6) / r

@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t compute_lj_potential(DTYPE_t r, DTYPE_t sigma, DTYPE_t epsilon, DTYPE_t rcutoff) nogil:
    if r >= rcutoff:
        return 0.0
    cdef DTYPE_t inv_r = sigma / r
    cdef DTYPE_t inv_r6 = pow(inv_r, 6)
    cdef DTYPE_t inv_r12 = inv_r6 * inv_r6
    cdef DTYPE_t pot = 4 * epsilon * (inv_r12 - inv_r6)
    cdef DTYPE_t inv_rcut = sigma / rcutoff
    cdef DTYPE_t shift = 4 * epsilon * (pow(inv_rcut, 12) - pow(inv_rcut, 6))
    return pot - shift

@cython.boundscheck(False)
@cython.wraparound(False)
def create_lattice(int n_particles, DTYPE_t box_size, int dimensions):
    cdef int n_side = <int>ceil(pow(n_particles, 1.0 / dimensions))
    cdef DTYPE_t spacing = box_size / n_side
    cdef np.ndarray[DTYPE_t, ndim=2] positions = np.zeros((n_particles, dimensions), dtype=DTYPE)
    cdef int count = 0
    cdef tuple indices
    cdef int i, j, k
    cdef DTYPE_t noise
    
    for indices in np.ndindex(*([n_side] * dimensions)):
        if count < n_particles:
            for k in range(dimensions):
                positions[count, k] = (indices[k] + 0.5) * spacing
                noise = (<DTYPE_t>rand() / RAND_MAX - 0.5) * 0.1 * spacing
                positions[count, k] += noise
            count += 1
    
    return positions

@cython.boundscheck(False)
@cython.wraparound(False)
def initialize_velocities(int n_particles, int dimensions, DTYPE_t target_temp):
    cdef np.ndarray[DTYPE_t, ndim=2] velocities = np.random.normal(0, 1, size=(n_particles, dimensions))
    cdef int i, j
    cdef np.ndarray[DTYPE_t, ndim=1] mean_vel = np.zeros(dimensions, dtype=DTYPE)
    cdef DTYPE_t ke, scale
    cdef int dof = n_particles * dimensions
    
    # Remove center-of-mass velocity
    for j in range(dimensions):
        mean_vel[j] = 0.0
        for i in range(n_particles):
            mean_vel[j] += velocities[i, j]
        mean_vel[j] /= n_particles
    
    for i in range(n_particles):
        for j in range(dimensions):
            velocities[i, j] -= mean_vel[j]
    
    # Scale to target temperature
    ke = 0.0
    for i in range(n_particles):
        for j in range(dimensions):
            ke += velocities[i, j] * velocities[i, j]
    ke *= 0.5
    
    scale = sqrt(target_temp * dof / (2 * ke))
    
    for i in range(n_particles):
        for j in range(dimensions):
            velocities[i, j] *= scale
    
    return velocities

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_berendsen_thermostat(np.ndarray[DTYPE_t, ndim=2] velocities, 
                              DTYPE_t target_temp, 
                              DTYPE_t current_temp, 
                              DTYPE_t dt, 
                              DTYPE_t tau):
    cdef DTYPE_t lambda_factor = sqrt(1 + dt / tau * (target_temp / current_temp - 1))
    cdef int i, j
    cdef int n = velocities.shape[0]
    cdef int d = velocities.shape[1]
    
    for i in range(n):
        for j in range(d):
            velocities[i, j] *= lambda_factor
    
    return velocities

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_langevin_thermostat(np.ndarray[DTYPE_t, ndim=2] velocities, 
                              DTYPE_t friction_coef, 
                              DTYPE_t dt, 
                              DTYPE_t temperature,
                              DTYPE_t mass,
                              DTYPE_t kb):
    cdef int n = velocities.shape[0]
    cdef int d = velocities.shape[1]
    cdef int i, j
    cdef DTYPE_t coeff, dissipative_vel, random_vel

    coeff = sqrt(2.0 * friction_coef * kb * temperature * dt) / mass
    cdef object normal_random = np.random.normal  
    for i in range(n):
        for j in range(d):
            # Dissipative term
            dissipative_vel = - (friction_coef / mass) * velocities[i, j] * dt
            # Random term (randn() is faster than np.random.normal)
            random_vel = coeff * normal_random(0,1)
            # Update
            velocities[i, j] += dissipative_vel + random_vel

    return velocities

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_forces_lca(np.ndarray[DTYPE_t, ndim=2] positions, 
                      DTYPE_t box_size, 
                      DTYPE_t rcutoff,
                      DTYPE_t sigma,
                      DTYPE_t epsilon,
                      bint use_pbc=True):
    
    cdef int n_particles = positions.shape[0]
    cdef int dim = positions.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] forces = np.zeros((n_particles, dim), dtype=DTYPE)
    cdef DTYPE_t potential_energy = 0.0
    
    cdef int i, j, k
    cdef DTYPE_t dist, f_mag
    cdef np.ndarray[DTYPE_t, ndim=1] r_ij = np.zeros(dim, dtype=DTYPE)
    
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            # Minimum image convention
            for k in range(dim):
                r_ij[k] = positions[i, k] - positions[j, k]
                if use_pbc:
                    r_ij[k] -= box_size * round(r_ij[k] / box_size)
            
            dist = 0.0
            for k in range(dim):
                dist += r_ij[k] * r_ij[k]
            dist = sqrt(dist)
            
            if dist < rcutoff and dist > 1e-12:
                f_mag = compute_lj_force(dist, sigma, epsilon, rcutoff)
                for k in range(dim):
                    forces[i, k] += f_mag * (r_ij[k] / dist)
                    forces[j, k] -= f_mag * (r_ij[k] / dist)
                potential_energy += compute_lj_potential(dist, sigma, epsilon, rcutoff)
    
    return forces, potential_energy

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_rdf(np.ndarray[DTYPE_t, ndim=2] positions, 
               DTYPE_t box_size, 
               DTYPE_t rcutoff,
               int n_bins=RDF_BINS):
    
    cdef int n_particles = positions.shape[0]
    cdef int dim = positions.shape[1]
    cdef DTYPE_t dr = rcutoff / n_bins
    cdef np.ndarray[DTYPE_t, ndim=1] hist = np.zeros(n_bins, dtype=DTYPE)
    
    cdef int i, j, k, bin_idx
    cdef DTYPE_t dist
    cdef np.ndarray[DTYPE_t, ndim=1] rij = np.zeros(dim, dtype=DTYPE)
    
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            # Minimum image convention
            for k in range(dim):
                rij[k] = positions[i, k] - positions[j, k]
                rij[k] -= box_size * round(rij[k] / box_size)
            
            dist = 0.0
            for k in range(dim):
                dist += rij[k] * rij[k]
            dist = sqrt(dist)
            
            if dist < rcutoff:
                bin_idx = <int>(dist / dr)
                hist[bin_idx] += 2  # Count each pair once
    
    # Normalization
    cdef np.ndarray[DTYPE_t, ndim=1] r = np.linspace(dr/2, rcutoff-dr/2, n_bins)
    cdef np.ndarray[DTYPE_t, ndim=1] shell_volumes
    cdef DTYPE_t density = n_particles / pow(box_size, dim)
    
    if dim == 2:
        shell_volumes = 2 * np.pi * r * dr  # Area of 2D annular shell
    else:
        shell_volumes = 4 * np.pi * r * r * dr  # Volume of 3D spherical shell
    
    cdef np.ndarray[DTYPE_t, ndim=1] g_r = hist / (shell_volumes * density * n_particles)
    
    return r, g_r

@cython.boundscheck(False)
@cython.wraparound(False)
def run_simulation(dict params, bint trajectory_save=False, str trajectory_file="trajectory.xyz", 
                  int save_freq=10, bint compute_rdf_flag=False, 
                  int rdf_last_steps=2000, int rdf_sample_freq=10):
    
    cdef DTYPE_t box_size = pow(params['n_particles'] / params['density'], 1.0 / params['dimensions'])
    cdef int n_particles = params['n_particles']
    cdef int dimensions = params['dimensions']
    cdef int steps = params['steps']
    cdef DTYPE_t dt = params['dt']
    
    # Initialize system
    cdef np.ndarray[DTYPE_t, ndim=2] positions = create_lattice(n_particles, box_size, dimensions)
    cdef np.ndarray[DTYPE_t, ndim=2] velocities = initialize_velocities(n_particles, dimensions, params['temperature'])
    cdef int dof = n_particles * dimensions
    
    # Results storage
    cdef np.ndarray[DTYPE_t, ndim=1] temp_list = np.zeros(steps, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] kin_energy_list = np.zeros(steps, dtype=DTYPE)  # New: Kinetic energy
    cdef np.ndarray[DTYPE_t, ndim=1] pot_energy_list = np.zeros(steps, dtype=DTYPE)  # New: Potential energy
    cdef np.ndarray[DTYPE_t, ndim=1] tot_energy_list = np.zeros(steps, dtype=DTYPE)  # New: Total energy
    cdef np.ndarray[DTYPE_t, ndim=1] rdf_accum = np.zeros(RDF_BINS, dtype=DTYPE)
    cdef int rdf_count = 0
    cdef np.ndarray[DTYPE_t, ndim=1] r_vals = np.zeros(RDF_BINS, dtype=DTYPE)
    
    # Open trajectory file if needed
    cdef object traj_file = None
    if trajectory_save:
        traj_file = open(trajectory_file, 'w')

    cdef np.ndarray[DTYPE_t, ndim=2] forces
    cdef DTYPE_t kinetic, temp, potential_energy  # New: Track potential energy
    cdef int step, i, j
    
    try:
        for step in range(steps):
            # Force calculation (now captures potential_energy)
            forces, potential_energy = compute_forces_lca(
                positions, box_size, params['rcutoff'],
                params['sigma'], params['epsilon'], True
            )
            
            # Velocity Verlet integration
            for i in range(n_particles):
                for j in range(dimensions):
                    velocities[i, j] += 0.5 * forces[i, j] * dt
                    positions[i, j] += velocities[i, j] * dt
                    positions[i, j] = fmod(positions[i, j], box_size)
            
            # Recompute forces after position update
            forces, potential_energy = compute_forces_lca(
                positions, box_size, params['rcutoff'],
                params['sigma'], params['epsilon'], True
            )
            
            for i in range(n_particles):
                for j in range(dimensions):
                    velocities[i, j] += 0.5 * forces[i, j] * dt
            
            # Compute kinetic energy and temperature
            kinetic = 0.0
            for i in range(n_particles):
                for j in range(dimensions):
                    kinetic += velocities[i, j] * velocities[i, j]
            kinetic *= 0.5
            temp = 2 * kinetic / dof
            temp_list[step] = temp
            
            # Store energies
            kin_energy_list[step] = kinetic
            pot_energy_list[step] = potential_energy
            tot_energy_list[step] = kinetic + potential_energy  # Total energy
            
            # Apply thermostat (if needed)
            if params['thermostat_type'] == 'berendsen':
                velocities = apply_berendsen_thermostat(velocities, params['temperature'], temp, dt, params['tau_ber'])
            elif params['thermostat_type'] == 'langevin':
                velocities = apply_langevin_thermostat(velocities, params['friction_coef'], params['dt'], params['temperature'],params['mass'],params['kb'])
            
            # Save trajectory (optional)
            if trajectory_save and (step % save_freq == 0):
                traj_file.write(f"{n_particles}\n")
                traj_file.write(f"Step {step}\n")
                for pos in positions:
                    if dimensions == 2:
                        traj_file.write(f"X {pos[0]} {pos[1]} 0.0\n")
                    else:
                        traj_file.write(f"X {pos[0]} {pos[1]} {pos[2]}\n")
            
            # RDF calculation (optional)
            if compute_rdf_flag and step >= (steps - rdf_last_steps) and (step % rdf_sample_freq == 0):
                r, gr = compute_rdf(positions, box_size, params['rcutoff'])
                r_vals = r
                for i in range(RDF_BINS):
                    rdf_accum[i] += gr[i]
                rdf_count += 1
    
    finally:
        if trajectory_save and traj_file is not None:
            traj_file.close()
    
    # Average RDF (if computed)
    cdef np.ndarray[DTYPE_t, ndim=1] rdf_avg = np.zeros(RDF_BINS, dtype=DTYPE)
    if rdf_count > 0:
        for i in range(RDF_BINS):
            rdf_avg[i] = rdf_accum[i] / rdf_count
    
    return {
        'temperature': temp_list,
        'kinetic_energy': kin_energy_list,  # New: Kinetic energy
        'potential_energy': pot_energy_list,  # New: Potential energy
        'total_energy': tot_energy_list,  # New: Total energy
        'rdf': (r_vals, rdf_avg) if compute_rdf_flag else (None, None),
        'positions': positions,
        'velocities': velocities,
        'trajectory_file': trajectory_file if trajectory_save else None
    }

# Python wrapper
def run_simulation_py(n_particles=100, dimensions=2, density=0.8, 
                     temperature=1.0, dt=0.005, steps=10000, 
                     rcutoff=2.5, sigma=1.0, epsilon=1.0,
                     tau_ber=0.1, friction_coef=1.0, 
                     thermostat_type='langevin',
                     trajectory_save=False,
                     trajectory_file="trajectory.xyz",
                     save_freq=10,
                     compute_rdf_flag=False,
                     rdf_last_steps=2000,
                     rdf_sample_freq=10,
                     mass=1,
                     kb=1
                     ):
    """
    Python wrapper for the Cython MD simulation.
    
    Parameters:
        trajectory_save: If True, saves trajectory to XYZ file
        trajectory_file: Output XYZ file path
        save_freq: Frequency (in steps) to save trajectory frames
        Other parameters same as before
    """
    params = {
        'n_particles': n_particles,
        'dimensions': dimensions,
        'density': density,
        'temperature': temperature,
        'dt': dt,
        'steps': steps,
        'rcutoff': rcutoff,
        'sigma': sigma,
        'epsilon': epsilon,
        'tau_ber': tau_ber,
        'friction_coef': friction_coef,
        'thermostat_type': thermostat_type,
        'mass': mass,
        'kb': kb
    }
    
    return run_simulation(
        params,
        trajectory_save=trajectory_save,
        trajectory_file=trajectory_file,
        save_freq=save_freq,
        compute_rdf_flag=compute_rdf_flag,
        rdf_last_steps=rdf_last_steps,
        rdf_sample_freq=rdf_sample_freq
    )