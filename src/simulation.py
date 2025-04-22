import numpy as np
import time
import os
from config import Configuration
from output_and_plots import save_xyz
from forces import compute_forces_virial
from thermostats import langevin_thermostat, berendsen_thermostat, nose_hoover_thermostat, parrinello_rahman_barostat

class Simulation:
    def __init__(self, config: Configuration):
        self.config = config
        self._load_config_params()
        self._initialize_simulation_state()
        self._initialize_output_files()

    def _load_config_params(self):
        c = self.config
        self.n_particles = c.n_particles
        self.dt = c.dt
        self.steps = c.steps
        self.temperature = c.temperature
        self.sigma = c.sigma
        self.epsilon = c.epsilon
        self.mass = c.mass
        self.kb = c.kb
        self.rcutoff = c.rcutoff
        self.use_langevin = c.use_langevin
        self.use_berendsen = c.use_berendsen
        self.thermostat_constant = c.thermostat_constant
        self.use_npt = c.use_npt
        self.target_pressure = c.target_pressure
        self.nh_Q = c.nh_Q
        self.pr_W = c.pr_W
        self.zeta = 0.0
        self.box_size = self.config.box_size

    def _initialize_simulation_state(self):
        if self.use_npt:
            self.positions = self.create_lattice()
        else:
            self.positions = self.create_lattice()
            # self.positions = self.read_xyz("npt_final.xyz")

        self.velocities, self.kinetic_energy = self.initialize_velocities()
        self.forces, self.potential_energy, self.virial_sum = compute_forces_virial(
            self.positions, self.config.box_size, self.rcutoff, self.sigma, self.epsilon
        )
        self.total_energy = self.kinetic_energy + self.potential_energy
        self.volume = self.config.volume
        self.box_size = self.config.box_size
        self.num_density = self.config.num_density
        self.density = self.config.density

    def _initialize_output_files(self):
        os.makedirs("output", exist_ok=True)
        self.trajectory_file = os.path.join("output", "trajectory.xyz")
        self.pressure_file = os.path.join("output", "pressure.dat")
        self.energy_file = os.path.join("output", "energy.dat")
        with open(self.trajectory_file, "w"), open(self.pressure_file, "w") as pf, open(self.energy_file, "w") as ef:
            pf.write("# Step Time(s) Pxy(Pa) Pxz(Pa) Pyz(Pa)\n")
            ef.write("# Step Time(s) KE(J) PE(J) TotalE(J) Temp(K) Volume\n")
        if self.use_npt:
            self.npt_final = os.path.join("output", "npt_final.xyz")

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

    def apply_boundary_conditions(self, positions):
        """Apply periodic boundary conditions using scalar box size."""
        # Use periodic boundary conditions
        positions %= self.box_size
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

    def compute_pressure(self):
        kinetic_pressure = (2.0 / (3.0 * self.volume)) * self.kinetic_energy
        # Virial is sum of diagonal components
        virial_trace = np.sum(self.virial_sum[:3])
        virial_pressure = virial_trace / (3.0 * self.volume)
        return kinetic_pressure + virial_pressure

    def velocity_verlet_step(self):
        """Perform one step of Velocity Verlet integration"""
        # Half-step velocity update
        self.velocities += 0.5 * self.dt * self.forces / self.mass
        # Apply thermostat if selected
        if not self.use_npt:
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
        self.forces, self.potential_energy, self.virial_sum = compute_forces_virial(
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
            self.forces, self.potential_energy, self.virial_sum = compute_forces_virial(
                self.positions, self.box_size, self.rcutoff,
                self.sigma, self.epsilon)
            # 2nd half-step velocity update
            self.velocities += 0.5 * self.dt * self.forces / self.mass
            # Compute energies and pressure
            self.kinetic_energy = 0.5 * self.mass * np.sum(self.velocities**2)
            self.total_energy = self.kinetic_energy + self.potential_energy
            self.update_pressure_tensor()
            return self.kinetic_energy, self.potential_energy, self.total_energy, self.pressure_tensor[3:]


    def minimize_energy_steepest_descent(self, max_steps=5000, force_tol=1e-6, step_size=0.01):
        print("\nStarting energy minimization (steepest descent)...")
        print(f"Convergence criterion: max force < {force_tol:.1e} N")

        # Store original positions to avoid modifying self.positions directly
        minimized_positions = self.positions.copy()
        converged = False

        for n_steps in range(max_steps):
            # Compute forces and potential energy
            forces, potential_energy, _ = compute_forces_virial(
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
            new_forces, new_pe, _ = compute_forces_virial(
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
                current_pressure = self.compute_pressure()
                print(f"Min step {n_steps}: PE = {potential_energy:.3e} J, max F = {max_force:.3e} N, step size = {step_size:.3f}, pressure = {current_pressure:.3f}")
        if not converged:
            print(f"Warning: Minimization did not converge after {max_steps} steps (max F = {max_force:.3e} N)")
        return minimized_positions


    def simulate_LJ(self):
        print("Starting simulation...")
        save_xyz(self.positions, self.trajectory_file, 0)
        start_time = time.time()

        if self.use_npt:
            minimized_positions = self.minimize_energy_steepest_descent()
            self.positions = minimized_positions

        for step in range(self.steps + 1):
            if self.use_npt:
                ke, pe, te, pt_offdiag = self.npt_step()
            else:
                ke, pe, te, pt_offdiag = self.velocity_verlet_step()

            t_sim = step * self.dt
            temp = ke / (1.5 * self.n_particles * self.kb)
            volume_str = f" {self.volume:.6e}" if self.use_npt else ""

            if step % 100 == 0:
                with open(self.energy_file, "a") as ef:
                    ef.write(f"{step} {t_sim:.6e} {ke:.6e} {pe:.6e} {te:.6e} {temp:.3f}{volume_str}\n")
            if step % 10 == 0:
                save_xyz(self.positions, self.trajectory_file, step)
            with open(self.pressure_file, "a") as pf:
                pf.write(f"{step} {t_sim:.6e} {pt_offdiag[0]:.6e} {pt_offdiag[1]:.6e} {pt_offdiag[2]:.6e}\n")
            if step % 100 == 0:
                current_pressure = self.compute_pressure()
                print(f"Step: {step:6d} | Time: {t_sim:8.3e} s | Temp: {temp:7.2f} K | Pressure = {current_pressure:.3f} Pa | Pxy: {pt_offdiag[0]:.3e} Pa")
            # Save the last frame to a seprate file npt_final.xyz
            if step == self.steps and self.use_npt:
                save_xyz(self.positions, self.npt_final, step)

        elapsed = time.time() - start_time
        print("\nSimulation finished.")
        print(f"Total steps: {self.steps}, Total sim time: {self.steps * self.dt:.2e} s")
        print(f"Wall time: {elapsed:.2f} s")
        if self.use_npt:
            print(f"Files saved: {self.trajectory_file}, {self.energy_file}, {self.pressure_file}, {self.npt_final}")
        else:
            print(f"Files saved: {self.trajectory_file}, {self.energy_file}, {self.pressure_file}")