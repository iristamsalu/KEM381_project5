from config import parse_args
from simulation import Simulation

if __name__ == "__main__":
    # Parse command-line arguments and create a config object
    config = parse_args()
    # Initialize the simulation with the configuration object
    sim = Simulation(config)
    # Run the simulation
    sim.simulate_LJ()

