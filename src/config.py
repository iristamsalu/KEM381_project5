import argparse
import sys
import os
from dataclasses import dataclass, field

# Physical constants and defaults
ARGON_SIGMA = 3.405e-10  # meters
ARGON_EPSILON_KB = 119.8  # K
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
ARGON_EPSILON = ARGON_EPSILON_KB * BOLTZMANN_CONSTANT
ARGON_MASS_KG = 39.948 * 1.66054e-27  # kg
DEFAULT_TEMP_K = 223
DEFAULT_DENSITY = 3.51  # kg/m3
DEFAULT_TIMESTEP = 2e-15  # s
DEFAULT_RCUTOFF = 2.5 * ARGON_SIGMA
DEFAULT_BERENDSEN_TAU = 0.5e-12 # s
DEFAULT_TARGET_PRESSURE = 0.1628e6  # pressure for NPT ensemble, Pa
DEFAULT_NH_Q = 4e-24  # Default Nose-Hoover coupling constant for NPT
DEFAULT_PR_W = 5e-28  # Default Parrinello-Rahman coupling constant for NPT

@dataclass
class Configuration:
    steps: int
    dt: float
    n_particles: int
    temperature: float
    sigma: float
    epsilon: float
    density: float
    rcutoff: float
    use_langevin: bool
    use_berendsen: bool
    thermostat_constant: float
    use_npt: bool
    use_nvt: bool
    use_nve: bool
    target_pressure: float
    nh_Q: float
    pr_W: float
    volume: float = None
    init_config: str = None
    mass: float = ARGON_MASS_KG
    kb: float = BOLTZMANN_CONSTANT

    num_density: float = field(init=False)
    box_size: float = field(init=False)

    def __post_init__(self):
        if self.use_npt:
            self.num_density = self.density / self.mass
            self.volume = self.n_particles / self.num_density
        else:
            self.n_particles = round(self.density * self.volume / self.mass)
            self.num_density = self.n_particles / self.volume

        self.box_size = self.volume ** (1 / 3)
        if self.rcutoff > self.box_size / 2:
            print("Error: Cutoff radius is too large for the box size.")
            sys.exit(1)

def validate_positive(value, name):
    if value <= 0:
        print(f"Error: {name} must be positive.")
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Lennard-Jones MD simulation with Argon (real units)")

    parser.add_argument("--steps", type=int, default=20000, help="Number of simulation steps")
    parser.add_argument("--dt", type=float, default=DEFAULT_TIMESTEP, help="Timestep in seconds")
    parser.add_argument("--n_particles", type=int, default=2000, help="Number of particles")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMP_K, help="Temperature in K")
    parser.add_argument("--density", type=float, default=DEFAULT_DENSITY, help="Mass density in kg/m3")
    parser.add_argument("--sigma", type=float, default=ARGON_SIGMA, help="Lennard-Jones sigma (m)")
    parser.add_argument("--epsilon", type=float, default=ARGON_EPSILON, help="Lennard-Jones epsilon (J)")
    parser.add_argument("--mass", type=float, default=ARGON_MASS_KG, help="Particle mass (kg)")
    parser.add_argument("--rcutoff", type=float, default=DEFAULT_RCUTOFF, help="Cutoff radius (m)")
    thermo_group = parser.add_mutually_exclusive_group()
    thermo_group.add_argument("--use_langevin", action="store_true", help="Use Langevin thermostat")
    thermo_group.add_argument("--use_berendsen", action="store_true", help="Use Berendsen thermostat")
    parser.add_argument("--thermostat_constant", type=float, default=None, help="Thermostat constant (Langevin: gamma, Berendsen: tau)")
    ensemble_group = parser.add_mutually_exclusive_group()
    ensemble_group.add_argument("--npt", action="store_true", help="Run in NPT ensemble")
    ensemble_group.add_argument("--nvt", action="store_true", help="Run in NVT ensemble")
    ensemble_group.add_argument("--nve", action="store_true", help="Run in NVE ensemble (no thermostat/barostat)")
    parser.add_argument("--volume", type=float, default=None, help="Box volume (for NVT/NVE)")
    parser.add_argument("--init_config", type=str, default=None, help="Initial config file (XYZ)")
    parser.add_argument("--target_pressure", type=float, default=DEFAULT_TARGET_PRESSURE, help="Target pressure for NPT (Pa)")
    parser.add_argument("--nh_Q", type=float, default=DEFAULT_NH_Q, help="Nose-Hoover coupling constant")
    parser.add_argument("--pr_W", type=float, default=DEFAULT_PR_W, help="Parrinello-Rahman coupling constant")

    args = parser.parse_args()

    if not args.npt and not args.nvt and not args.nve:
        print("Info: No ensemble specified. Defaulting to NVE (no thermostat, no barostat).")
        args.nve = True
    if args.thermostat_constant is None:
        if args.use_langevin:
            args.thermostat_constant = args.mass / DEFAULT_BERENDSEN_TAU
        elif args.use_berendsen:
            args.thermostat_constant = DEFAULT_BERENDSEN_TAU

    # Validate positive values
    for name in ["steps", "dt", "n_particles", "temperature", "density", "sigma", "epsilon", "mass", "rcutoff"]:
        validate_positive(getattr(args, name), name)

    if args.use_langevin or args.use_berendsen:
        validate_positive(args.thermostat_constant, "thermostat_constant")

    os.makedirs("output", exist_ok=True)

    config = Configuration(
        steps=args.steps,
        dt=args.dt,
        n_particles=args.n_particles,
        temperature=args.temperature,
        sigma=args.sigma,
        epsilon=args.epsilon,
        density=args.density,
        rcutoff=args.rcutoff,
        use_langevin=args.use_langevin,
        use_berendsen=args.use_berendsen,
        thermostat_constant=args.thermostat_constant,
        use_npt=args.npt,
        use_nvt=args.nvt,
        use_nve=args.nve,
        target_pressure=args.target_pressure,
        nh_Q=args.nh_Q,
        pr_W=args.pr_W,
        volume=args.volume,
        init_config=args.init_config,
        mass=args.mass,
    )

    return config