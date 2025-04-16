from config import parse_args
from simulation import Simulation

if __name__ == "__main__":
    # Parse command-line arguments and create a config object
    config = parse_args()
    # Initialize the simulation with the configuration object
    sim = Simulation(config)

    # Run the Lennard-Jones simulation
    if config.use_lca:
        # Run with the linked cell algorithm
        sim.simulate_LJ()
        print("Lennard-Jones simulation with linked cell algorithm is complete.")
    else:
        # Run with the naive algorithm
        sim.simulate_LJ()
        print("Lennard-Jones simulation with naive algorithm is complete.")
