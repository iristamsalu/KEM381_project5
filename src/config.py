import argparse
import sys
import os
from dataclasses import dataclass, field

# Define Argon physical constants
ARGON_SIGMA = 3.405e-10  # m
ARGON_EPSILON_KB = 119.8  # Epsilon divided by Boltzmann constant (K)
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
ARGON_EPSILON = ARGON_EPSILON_KB * BOLTZMANN_CONSTANT # J
ARGON_MASS_AMU = 39.948 # amu
AMU_TO_KG = 1.66054e-27 # kg/amu
ARGON_MASS_KG = ARGON_MASS_AMU * AMU_TO_KG # kg

# Change depending on the desired system
DEFAULT_TEMP_K = 107.7 # K
DEFAULT_DENSITY_KG_M3 = 1296.2 # kg/m3 (90K, 1atm)
DEFAULT_TIMESTEP_S = 2e-15 # 5 femtoseconds (s)
DEFAULT_RCUTOFF_FACTOR = 2.5 # Cutoff radius as a multiple of sigma
DEFAULT_RCUTOFF_M = DEFAULT_RCUTOFF_FACTOR * ARGON_SIGMA # m

# Default Thermostat constants in real units
DEFAULT_BERENDSEN_TAU_S = 0.5e-12 # 0.5 picoseconds (s)
DEFAULT_LANGEVIN_GAMMA_KGS = ARGON_MASS_KG / DEFAULT_BERENDSEN_TAU_S # kg/s
# Add these near other default constants
DEFAULT_TARGET_PRESSURE_PA = 8.934100e5  # in Pascals
DEFAULT_NH_Q = 0.001  # Default Nose-Hoover coupling constant
DEFAULT_PR_W = 0.005  # Default Parrinello-Rahman coupling constant

@dataclass
class Configuration:
    # Core simulation parameters
    steps: int
    dt: float           # timestep in seconds
    n_particles: int
    temperature: float  # temperature in K

    # Particle properties in real units
    sigma: float                        # LJ sigma in m
    epsilon: float                      # LJ epsilon in J

    # System properties in real units
    density: float      # mass density in kg/m3
    rcutoff: float      # cutoff radius in m

    # Derived properties added after parsing
    num_density: float = field(init=False) # Number density in particles/m3
    box_size: float = field(init=False)    # Box edge length in m
    volume: float = field(init=False)      # Box volume in m^3

    # Algorithm flag
    use_lca: bool

    # Thermostats and parameters
    use_langevin: bool
    use_berendsen: bool
    thermostat_constant: float  # gamma (kg/s) for Langevin, tau (s) for Berendsen

    # Default parameters
    mass: float = ARGON_MASS_KG         # Particle mass in kg
    kb: float = BOLTZMANN_CONSTANT      # Boltzmann constant in J/K
    # NPT parameters
    use_npt: bool = False  # Enable/disable NPT ensemble
    target_pressure: float = DEFAULT_TARGET_PRESSURE_PA  # Target pressure in Pa
    nh_Q: float = DEFAULT_NH_Q  # Nosé-Hoover coupling constant
    pr_W: float = DEFAULT_PR_W  # Parrinello-Rahman coupling constant
    
    def __post_init__(self):
        """Calculate derived properties after arguments are parsed."""
        if self.mass <= 0:
             print("Error: Particle mass must be positive.")
             sys.exit(1)
        if self.density <= 0:
             print("Error: Density must be positive.")
             sys.exit(1)
             
        self.num_density = self.density / self.mass # particles/m3
        if self.num_density <= 0 or self.n_particles <=0:
             print("Error: Invalid number density or particle count.")
             sys.exit(1)
             
        self.volume = self.n_particles / self.num_density # m3
        self.box_size = self.volume ** (1 / 3) # side length in m

        # Validate rcutoff against box size
        if self.rcutoff > self.box_size / 2.0:
             print(f"Error: Cutoff radius ({self.rcutoff:.3e} m) cannot be larger than half the box size ({self.box_size/2.0:.3e} m) when using PBC.")
             sys.exit(1)
             
        # Add a note about the thermostat
        if self.use_langevin:
            print(f"Info: Using Langevin thermostat with gamma = {self.thermostat_constant:.3e} kg/s")
        elif self.use_berendsen:
            print(f"Info: Using Berendsen thermostat with tau = {self.thermostat_constant:.3e} s")

def validate_positive(value, name):
    """Check if a value is positive."""
    if value <= 0:
        print(f"Error: {name} must be greater than 0.")
        sys.exit(1)

def parse_args():
    """Parse and validate the command line arguments."""
    parser = argparse.ArgumentParser(description="Lennard-Jones MD simulation in real units (default: Argon)")
    
    # Arguments for the simulation
    parser.add_argument("--steps", type=int, default=100000, help="Number of simulation steps")
    parser.add_argument("--dt", type=float, default=DEFAULT_TIMESTEP_S, help=f"Timestep in seconds (Default: {DEFAULT_TIMESTEP_S:.1e} s)")
    parser.add_argument("--density", type=float, default=DEFAULT_DENSITY_KG_M3, help=f"Mass density in kg/m3 (Default: {DEFAULT_DENSITY_KG_M3:.1f} kg/m3)")
    parser.add_argument("--n_particles", type=int, default=1024, help="Number of particles (Default: 1024)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMP_K, help=f"Desired temperature in Kelvin (Default: {DEFAULT_TEMP_K:.1f} K)")
    # Allow overriding LJ parameters, but default to Argon
    parser.add_argument("--sigma", type=float, default=ARGON_SIGMA, help=f"Lennard-Jones sigma parameter in meters (Default: {ARGON_SIGMA:.3e} m)")
    parser.add_argument("--epsilon", type=float, default=ARGON_EPSILON, help=f"Lennard-Jones epsilon parameter in Joules (Default: {ARGON_EPSILON:.3e} J)")
    parser.add_argument("--mass", type=float, default=ARGON_MASS_KG, help=f"Particle mass in kilograms (Default: {ARGON_MASS_KG:.3e} kg)")
    parser.add_argument("--rcutoff", type=float, default=DEFAULT_RCUTOFF_M, help=f"Lennard-Jones cutoff radius in meters (Default: {DEFAULT_RCUTOFF_M:.3e} m)")
    # Algorithm flag
    parser.add_argument("--use_lca", action="store_true", help="Use Linked Cell Algorithm (LCA)")
    # Thermostat selection
    thermo_group = parser.add_mutually_exclusive_group() # user can't pick both
    thermo_group.add_argument("--use_langevin", action="store_true", help="Use Langevin thermostat")
    thermo_group.add_argument("--use_berendsen", action="store_true", help="Use Berendsen thermostat")
    parser.add_argument("--thermostat_constant", type=float, default=None, help="Thermostat constant (Langevin: gamma in kg/s, Berendsen: tau in s)")
    # Add these new arguments for NPT
    parser.add_argument("--use_npt", action="store_true", help="Enable NPT ensemble (constant pressure)")
    parser.add_argument("--target_pressure", type=float, default=DEFAULT_TARGET_PRESSURE_PA,
                       help=f"Target pressure in Pascals (Default: {DEFAULT_TARGET_PRESSURE_PA:.3e} Pa = 1 atm)")
    parser.add_argument("--nh_Q", type=float, default=DEFAULT_NH_Q, help=f"Nosé-Hoover coupling constant (Default: {DEFAULT_NH_Q:.1f})")
    parser.add_argument("--pr_W", type=float, default=DEFAULT_PR_W, help=f"Parrinello-Rahman coupling constant (Default: {DEFAULT_PR_W:.1f})")
    args = parser.parse_args()

    # Set a default thermostat constant
    if args.thermostat_constant is None:
        if args.use_langevin:
             # Calculate default gamma using the parsed mass
            default_langevin_gamma = args.mass / DEFAULT_BERENDSEN_TAU_S
            args.thermostat_constant = default_langevin_gamma
            print(f"Info: Using default Langevin gamma = {args.thermostat_constant:.3e} kg/s (based on mass)")
        elif args.use_berendsen:
            args.thermostat_constant = DEFAULT_BERENDSEN_TAU_S
            print(f"Info: Using default Berendsen tau = {args.thermostat_constant:.3e} s")
        # Additional validation for NPT parameters
        elif args.use_npt:
            print("Warning: NPT ensemble uses Nosé-Hoover thermostat - Langevin/Berendsen flags will be ignored")
            args.use_langevin = False
            args.use_berendsen = False
            validate_positive(args.target_pressure, "Target pressure")
            validate_positive(args.nh_Q, "Nose-Hoover Q")
            validate_positive(args.pr_W, "Parrinello-Rahman W")
            print(f"Info: Using NPT ensemble with target pressure = {args.target_pressure:.3e} Pa")
        else:
             # Default to Langevin if neither specified
             args.use_langevin = True
             default_langevin_gamma = args.mass / DEFAULT_BERENDSEN_TAU_S
             args.thermostat_constant = default_langevin_gamma
             print(f"Info: No thermostat specified, defaulting to Langevin with gamma = {args.thermostat_constant:.3e} kg/s (based on mass)")

    # Validate the input
    validate_positive(args.steps, "Number of steps")
    validate_positive(args.dt, "Time step")
    validate_positive(args.density, "Density")
    validate_positive(args.n_particles, "Number of particles")
    validate_positive(args.temperature, "Temperature")
    validate_positive(args.sigma, "Sigma")
    validate_positive(args.epsilon, "Epsilon")
    validate_positive(args.mass, "Mass")
    validate_positive(args.rcutoff, "Cutoff radius")
    if args.use_langevin or args.use_berendsen:
         validate_positive(args.thermostat_constant, "Thermostat constant")

    # Create output folder
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)  

    # Create Configuration
    config = Configuration(
        steps=args.steps,
        dt=args.dt,
        density=args.density,
        n_particles=args.n_particles,
        temperature=args.temperature,
        sigma=args.sigma,
        epsilon=args.epsilon,
        mass=args.mass,
        rcutoff=args.rcutoff,
        use_lca=args.use_lca,
        use_langevin=args.use_langevin,
        use_berendsen=args.use_berendsen,
        thermostat_constant=args.thermostat_constant,
        use_npt=args.use_npt,
        target_pressure=args.target_pressure,
        nh_Q=args.nh_Q,
        pr_W=args.pr_W,
    )

    return config
