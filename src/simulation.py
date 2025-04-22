import numpy as np
import time
import os
from config import Configuration
from output_and_plots import save_xyz
from forces import compute_forces_lca_virial, compute_forces_naive_virial
from thermostats import langevin_thermostat, berendsen_thermostat, nose_hoover_thermostat, parrinello_rahman_barostat

class Simulation:
    def __init__(self, config: Configuration):
        """Initialize the simulation with a configuration object."""
        self.config = config
        # Core parameters
        self.n_particles = config.n_particles
        self.dt = config.dt
        self.steps = config.steps
        self.temperature = config.temperature
        # Particle properties
        self.sigma = config.sigma       # m
        self.epsilon = config.epsilon   # J
        self.mass = config.mass         # kg
        self.kb = config.kb             # J/K
        # System properties
        self.density = config.density     # kg/m^3
        self.rcutoff = config.rcutoff     # m
        self.num_density = config.num_density # particles/m^3
        self.box_size = config.box_size       # m (scalar length)
        self.volume = config.volume           # m^3
        # Algorithm choice
        self.use_lca = config.use_lca
        # Thermostats
        self.use_langevin = config.use_langevin
        self.use_berendsen = config.use_berendsen
        self.thermostat_constant = config.thermostat_constant
        # NPT parameters
        self.use_npt = config.use_npt  # Boolean to enable/disable NPT
        self.target_pressure = config.target_pressure  # Target pressure in Pa
        self.nh_Q = config.nh_Q  # Nosé-Hoover coupling constant
        self.pr_W = config.pr_W  # Parrinello-Rahman coupling constant
        self.zeta = 0.0  # Initialize thermostat variable

        # Create initial 3D lattice (m)
        self.positions = self.create_lattice()
        # self.positions = self.read_xyz("npt_final.xyz")
        # print(f"Initial positions loaded from file: {self.positions}")
        # Initialize velocities (m/s) randomly and adjust to desired temperature
        self.velocities, self.kinetic_energy = self.initialize_velocities()

        # Choose the force and virial computation method for the simulation
        if self.use_lca:
            print("Using Linked Cell Algorithm (LCA).")
            self.compute_forces_virial = compute_forces_lca_virial
        else:
            print("Using naive force calculation.")
            self.compute_forces_virial = compute_forces_naive_virial

        # Initialize forces (N), potential energy (J), and virial sum (J)
        print("Calculating initial forces and virial...")
        self.forces, self.potential_energy, self.virial_sum = self.compute_forces_virial(
            self.positions, self.box_size, self.rcutoff,
            self.sigma, self.epsilon
        )

        # Calculate total energy
        self.total_energy = self.kinetic_energy + self.potential_energy

        print(f"Initial KE: {self.kinetic_energy:.3e} J, PE: {self.potential_energy:.3e} J, Total E: {self.total_energy:.3e} J")
        initial_temp = self.kinetic_energy / (1.5 * self.n_particles * self.kb) # 3D DOF = 3N -> KE = 1.5 N kb T
        print(f"Initial Temp: {initial_temp:.2f} K")

        # Setup output file paths
        self.trajectory_file = os.path.join("output", "trajectory.xyz")
        self.pressure_file = os.path.join("output", "pressure.dat")
        self.energy_file = os.path.join("output", "energy.dat") 

    def create_lattice(self):
        """Create a 3D simple cubic lattice with slight random displacements."""
        n_side = int(np.ceil(self.n_particles ** (1 / 3)))
        if n_side**3 < self.n_particles: n_side += 1 
        spacing = self.box_size / n_side 
        positions = []
        for i in range(n_side):
            for j in range(n_side):
                for k in range(n_side):
                    if len(positions) < self.n_particles:
                        pos = np.array([(i + 0.5)*spacing, (j + 0.5)*spacing, (k + 0.5)*spacing])
                        noise_scale = 0.05 
                        noise = (np.random.rand(3) - 0.5) * spacing * noise_scale 
                        positions.append(pos + noise)
        positions = np.array(positions)
        positions %= self.box_size 
        if len(positions) != self.n_particles:
             positions = positions[:self.n_particles] 
        return positions

    def minimize_energy_steepest_descent(self, max_steps=5000, force_tol=1e-6, step_size=0.01):
        print("\nStarting energy minimization (steepest descent)...")
        print(f"Convergence criterion: max force < {force_tol:.1e} N")

        # Store original positions to avoid modifying self.positions directly
        minimized_positions = self.positions.copy()
        converged = False

        for n_steps in range(max_steps):
            # Compute forces and potential energy
            forces, potential_energy, _ = self.compute_forces_virial(
                minimized_positions, self.box_size, self.rcutoff,
                self.sigma, self.epsilon
            )

            # Check convergence
            max_force = np.max(np.abs(forces))
            if max_force < force_tol:
                converged = True
                print(f"Converged after {n_steps} steps with max force = {max_force:.3e} N")
                break
            
            # Steepest descent step: r_new = r_old + step_size * F/m (displace along forces)
            displacement = step_size * forces / self.mass
            minimized_positions += displacement

            # Apply periodic boundary conditions
            minimized_positions = self.apply_boundary_conditions(minimized_positions)

            # Adaptive step size - reduce if energy increases
            new_forces, new_pe, _ = self.compute_forces_virial(
                minimized_positions, self.box_size, self.rcutoff,
                self.sigma, self.epsilon
            )

            if new_pe > potential_energy:
                step_size *= 0.5  # Reduce step size if energy increased
                minimized_positions -= displacement  # Revert step
            else:
                step_size *= 1.05  # Slightly increase step size if energy decreased

            # Print progress
            if n_steps % 100 == 0:
                print(f"Min step {n_steps}: PE = {potential_energy:.3e} J, max F = {max_force:.3e} N, step size = {step_size:.3f}")

        if not converged:
            print(f"Warning: Minimization did not converge after {max_steps} steps (max F = {max_force:.3e} N)")

        return minimized_positions, n_steps, converged

    def initialize_velocities(self):
        """Generate initial velocities from Maxwell-Boltzmann distribution (m/s)."""
        # Generate small random initial velocities
        vel_std_dev = np.sqrt(self.kb * self.temperature / self.mass)
        velocities = np.random.normal(0.0, vel_std_dev, size=(self.n_particles, 3)) 
        velocities -= np.mean(velocities, axis=0)   
        current_ke = 0.5 * self.mass * np.sum(velocities**2)
        target_ke = 1.5 * self.n_particles * self.kb * self.temperature 
        scaling_factor = np.sqrt(target_ke / current_ke)
        velocities *= scaling_factor
        return velocities, target_ke
    
    def update_pressure_tensor(self):
        """Calculates the 6 components of the pressure tensor in Pascals (Pa)."""
        kinetic_tensor_sum = np.zeros(6) 
        vx, vy, vz = self.velocities[:, 0], self.velocities[:, 1], self.velocities[:, 2]
        kinetic_tensor_sum[0] = self.mass * np.sum(vx * vx)
        kinetic_tensor_sum[1] = self.mass * np.sum(vy * vy)
        kinetic_tensor_sum[2] = self.mass * np.sum(vz * vz) 
        kinetic_tensor_sum[3] = self.mass * np.sum(vx * vy)
        kinetic_tensor_sum[4] = self.mass * np.sum(vx * vz) 
        kinetic_tensor_sum[5] = self.mass * np.sum(vy * vz) 
        self.pressure_tensor = (kinetic_tensor_sum + self.virial_sum) / self.volume

    def velocity_verlet_step(self):
        """Perform one step of Velocity Verlet integration"""

        # Half-step velocity update
        self.velocities += 0.5 * self.dt * self.forces / self.mass

        # Apply thermostat if selected
        if self.use_langevin:
            self.velocities = langevin_thermostat(
                velocities=self.velocities, 
                dt=self.dt, 
                temperature=self.temperature, 
                friction_coef=self.thermostat_constant,
                mass=self.mass, 
                kb=self.kb)
        elif self.use_berendsen:
            self.velocities = berendsen_thermostat(
                velocities=self.velocities, 
                dt=self.dt, 
                target_temperature=self.temperature, 
                tau=self.thermostat_constant,
                n_particles=self.n_particles, 
                mass=self.mass, 
                kb=self.kb)

        # Update positions
        temp_positions = self.positions + self.velocities * self.dt
        self.positions = self.apply_boundary_conditions(temp_positions) 

        # Compute new forces
        self.forces, self.potential_energy, self.virial_sum = self.compute_forces_virial(
            self.positions, self.box_size, self.rcutoff,
            self.sigma, self.epsilon)

        # 2nd half-step velocity update
        self.velocities += 0.5 * self.dt * self.forces / self.mass 

        # Compute kinetic and total energy
        self.kinetic_energy = 0.5 * self.mass * np.sum(self.velocities**2)
        self.total_energy = self.kinetic_energy + self.potential_energy
        self.update_pressure_tensor() 

        # Return KE, PE, Total E (all in J), and off-diagonal PT (Pa)
        return self.kinetic_energy, self.potential_energy, self.total_energy, self.pressure_tensor[3:]

    def apply_boundary_conditions(self, positions):
        """Apply periodic boundary conditions using scalar box size."""
        # Use periodic boundary conditions
        positions %= self.box_size
        return positions

    def npt_step(self):
        """Perform one step of NPT integration using Nosé-Hoover and Parrinello-Rahman"""

        # Half-step velocity update
        self.velocities += 0.5 * self.dt * self.forces / self.mass

        # Apply Nose-Hoover thermostat
        if self.use_npt:
            self.velocities, self.zeta = nose_hoover_thermostat(
                velocities=self.velocities,
                dt=self.dt,
                temperature=self.temperature,
                kb=self.kb,
                mass=self.mass,
                Q=self.nh_Q,
                zeta=self.zeta
            )

        # Update positions
        temp_positions = self.positions + self.velocities * self.dt

        # Apply Parrinello-Rahman barostat
        if self.use_npt:
            (temp_positions, 
             self.velocities, 
             self.box_size, 
             scaling_factor) = parrinello_rahman_barostat(
                positions=temp_positions,
                velocities=self.velocities,
                forces=self.forces,
                dt=self.dt,
                pressure=self.target_pressure,
                W=self.pr_W,
                box_size=self.box_size,
                mass=self.mass  # Add this parameter
            )
            # Update system properties that depend on box size
            self.volume = self.box_size**3
            self.num_density = self.n_particles / self.volume
            self.density = self.num_density * self.mass

            self.positions = self.apply_boundary_conditions(temp_positions)

            # Compute new forces
            self.forces, self.potential_energy, self.virial_sum = self.compute_forces_virial(
                self.positions, self.box_size, self.rcutoff,
                self.sigma, self.epsilon)

            # 2nd half-step velocity update
            self.velocities += 0.5 * self.dt * self.forces / self.mass

            # Compute energies and pressure
            self.kinetic_energy = 0.5 * self.mass * np.sum(self.velocities**2)
            self.total_energy = self.kinetic_energy + self.potential_energy
            self.update_pressure_tensor()
            pressure = self.compute_pressure()
            return self.kinetic_energy, self.potential_energy, self.total_energy, self.pressure_tensor[3:], pressure

    def read_xyz(self, filename):
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        # Parse XYZ file
        n_atoms = int(lines[0])
        positions = []

        for line in lines[2:2+n_atoms]:  # Skip first 2 lines (atom count and comment)
            parts = line.split()
            # Convert Angstroms to meters if needed (detect by typical values)
            x, y, z = map(float, parts[1:4])
            if max(abs(x), abs(y), abs(z)) > 10:  # Likely in Ångstroms
                x, y, z = x*1e-10, y*1e-10, z*1e-10
            positions.append([x, y, z])

        positions = np.array(positions)

        positions %= self.box_size 

        # Verify atom count matches simulation
        if len(positions) != self.n_particles:
            raise ValueError(f"XYZ file contains {len(positions)} atoms but simulation expects {self.n_particles}")

        return positions
    def compute_pressure(self):
        kinetic_pressure = (2.0 / (3.0 * self.volume)) * self.kinetic_energy
        # Virial is sum of diagonal components
        virial_trace = np.sum(self.virial_sum[:3])
        virial_pressure = virial_trace / (3.0 * self.volume)
        return kinetic_pressure + virial_pressure
    
    def simulate_LJ(self):
        """Run Lennard-Jones simulation collecting pressure tensor data."""

        os.makedirs("output", exist_ok=True)
        with open(self.trajectory_file, "w") as f: pass 
        save_xyz(self.positions, self.trajectory_file, 0) 
        with open(self.pressure_file, "w") as pf:
            pf.write("# Step Time(s) Pxy(Pa) Pxz(Pa) Pyz(Pa)\n")
        with open(self.energy_file, "w") as ef:
             ef.write("# Step Time(s) KE(J) PE(J) TotalE(J) Temp(K) Volume Pressure\n")
      
        print("Starting simulation...")
        # Start tracking computational time
        start_time = time.time()
        # # Perform energy minimization before starting the simulation
        # minimized_positions, n_steps, converged = self.minimize_energy_steepest_descent()
        # if converged:
        #     self.positions = minimized_positions
        #     print(f"Minimized positions: {self.positions}")
        # else:
        #     print("Warning: Minimization did not converge. Continuing with initial positions.")
         
        # Simulation loop
        if self.use_npt:
            # Simulation loop
            for step in range(self.steps+1):
                kinetic_energy, potential_energy, total_energy, pt_offdiag, pressure = self.npt_step()
    
                current_sim_time = step * self.dt

                # Save pressure tensor at every step
                with open(self.pressure_file, "a") as pf:
                    pf.write(f"{step} {current_sim_time:.6e} {pt_offdiag[0]:.6e} {pt_offdiag[1]:.6e} {pt_offdiag[2]:.6e}\n")

                # Save energy/temp periodically
                if step % 100 == 0:
                     current_temp = kinetic_energy / (1.5 * self.n_particles * self.kb)
                     with open(self.energy_file, "a") as ef:
                          ef.write(f"{step} {current_sim_time:.6e} {kinetic_energy:.6e} {potential_energy:.6e} {total_energy:.6e} {current_temp:.3f} {self.volume:.6e} {pressure:.6e}\n")

                # Save positions periodically
                if step % 10 == 0:
                    save_xyz(self.positions, self.trajectory_file, step)

                # Print progress periodically
                if step % 10000 == 0:
                    current_temp = kinetic_energy / (1.5 * self.n_particles * self.kb) 
                    print(f"Step: {step:10d} | Time: {current_sim_time:8.3e} s | Temp: {current_temp:8.2f} K | Pxy: {pt_offdiag[0]:10.4e} Pa")

        else:
            for step in range(self.steps+1):
                kinetic_energy, potential_energy, total_energy, pt_offdiag = self.velocity_verlet_step()
                current_sim_time = step * self.dt

                # Save pressure tensor at every step
                with open(self.pressure_file, "a") as pf:
                    pf.write(f"{step} {current_sim_time:.6e} {pt_offdiag[0]:.6e} {pt_offdiag[1]:.6e} {pt_offdiag[2]:.6e}\n")

                # Save energy/temp periodically
                if step % 100 == 0:
                     current_temp = kinetic_energy / (1.5 * self.n_particles * self.kb)
                     with open(self.energy_file, "a") as ef:
                          ef.write(f"{step} {current_sim_time:.6e} {kinetic_energy:.6e} {potential_energy:.6e} {total_energy:.6e} {current_temp:.3f}\n")

                periodicity = 10
                # Save positions periodically
                if step % periodicity == 0:
                    save_xyz(self.positions, self.trajectory_file, step+periodicity)

                # Print progress periodically
                if step % 10000 == 0:
                    current_temp = kinetic_energy / (1.5 * self.n_particles * self.kb) 
                    print(f"Step: {step:10d} | Time: {current_sim_time:8.3e} s | Temp: {current_temp:8.2f} K | Pxy: {pt_offdiag[0]:10.4e} Pa")
        
        end_time = time.time()
        total_runtime = end_time - start_time
        final_sim_time = (self.steps) * self.dt
        print(f"\nSimulation finished.")
        print(f"Total steps simulated: {self.steps}")
        print(f"Total simulation time: {final_sim_time:.3e} s")
        print(f"Total computer time: {total_runtime:.2f} seconds")
        print(f"Pressure tensor data saved to: {self.pressure_file}")
        if os.path.exists(self.energy_file): print(f"Energy data saved to: {self.energy_file}")
        if os.path.exists(self.trajectory_file): print(f"Trajectory data saved to: {self.trajectory_file}")
            