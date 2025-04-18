import numpy as np
import time
import os
from config import Configuration
from output_and_plots import save_xyz
from forces import compute_forces_lca_virial, compute_forces_naive_virial
from thermostats import langevin_thermostat, berendsen_thermostat

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

        print("Initializing simulation (3D, Real Units)...")
        print(f"Box size: {self.box_size:.3e} m, Volume: {self.volume:.3e} m^3, Number density: {self.num_density:.3e} particles/m^3")

        # Create initial 3D lattice (m)
        self.positions = self.create_lattice()

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

    def simulate_LJ(self):
        """Run Lennard-Jones simulation collecting pressure tensor data."""

        os.makedirs("output", exist_ok=True)
        with open(self.trajectory_file, "w") as f: pass 
        save_xyz(self.positions, self.trajectory_file, 0) 
        with open(self.pressure_file, "w") as pf:
            pf.write("# Step Time(s) Pxy(Pa) Pxz(Pa) Pyz(Pa)\n")
        with open(self.energy_file, "w") as ef:
             ef.write("# Step Time(s) KE(J) PE(J) TotalE(J) Temp(K)\n")
      
        print("Starting simulation...")
        # Start tracking computational time
        start_time = time.time()

        # Simulation loop
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

            # Save positions periodically
            if step % 10 == 0:
                save_xyz(self.positions, self.trajectory_file, step)

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
            