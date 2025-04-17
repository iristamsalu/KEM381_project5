import matplotlib.pyplot as plt
from md_sim import run_simulation_py
import matplotlib.pyplot as plt

# Save 
if __name__ == "__main__":
    # Simulation parameters
    # n_particles = 100
    # dimensions = 3
    # dt = 0.001
    # # density = 0.4
    # # densities = [0.1,0.4,1]
    # # T=0.1
    # # Run simulation
    # # temperature = [0.8,0.6,0.4,0.3,0.1]
    # # for density in densities:
    # density = 0.03
    # steps = 100000

    # Run with trajectory saving


    # # Run with XYZ trajectory saving
    # results = run_simulation_py(
    #     steps=steps,
    #     trajectory_save=True,
    #     trajectory_file="my_traj.xyz",
    #     n_particles=n_particles,
    #     dimensions=dimensions,
    #     density=density,
    #     temperature=T,
    #     save_freq=5  # Save every 5 steps
    #     )

            # Plot temperature
        # plt.figure(figsize=(12, 4))
        # plt.subplot(1, 2, 1)
        # plt.plot(results['temperature'])
        # plt.xlabel('Step')
        # plt.ylabel('Temperature')
        # plt.title('Temperature Evolution')
        # Plot RDF
        # plt.subplot(1, 2, 2)
    #     r, g_r = results['rdf']
    #     positions = results['positions']
        
    #     plt.plot(r, g_r,label=f'g(r) {density}')
    #     plt.xlabel('r')
    #     plt.ylabel('g(r)')
    #     plt.title('Radial Distribution Function (Last 2000 steps)')
    #     plt.tight_layout()
    #     plt.grid(True)
    #     plt.legend()
    # plt.savefig(f'test_densities/md_res_all.png')
    #     # Plot RDF
    #     r, g_r = results['rdf']
    #     # positions = results['positions']
    #     plt.plot(r, g_r,label=f'g(r) Temperature= {T}')
    #     plt.xlabel('r')
    #     plt.ylabel('g(r)')
    #     plt.title('Radial Distribution Function (Last 2000 steps)')
    #     plt.tight_layout()
    #     plt.grid(True)
    #     plt.legend()
    #     plt.savefig(f'test_temp/md_res_{T}.png')
    #     plt.clf()
    # T=0.4
    n_particles=100
    dimensions=3 
    density=0.8 
    temperature=1.0
    dt=0.005
    steps=10000 
    rcutoff=2.5
    sigma=1.0
    epsilon=1.0
    tau_ber=0.1
    friction_coef=1.0 
    thermostat_type='langevin'
    trajectory_save=True
    trajectory_file="trajectory.xyz"
    save_freq=10
    compute_rdf_flag=True
    rdf_last_steps=2000
    rdf_sample_freq=10
    mass=1
    kb=1
    results = run_simulation_py(
        n_particles=n_particles,dimensions=dimensions, density=density, temperature=temperature, dt=dt, steps=steps, rcutoff=rcutoff, 
        sigma=sigma, epsilon=epsilon, tau_ber=tau_ber, friction_coef=friction_coef, thermostat_type=thermostat_type, 
        trajectory_save=trajectory_save, trajectory_file=trajectory_file, save_freq=save_freq, compute_rdf_flag=compute_rdf_flag, 
        rdf_last_steps=rdf_last_steps, rdf_sample_freq=rdf_sample_freq,mass=mass, kb=kb
        )
    # Plot Energy
    plt.figure(figsize=(12, 4))
    plt.plot(results['total_energy'])
    plt.plot(results['potential_energy'])
    plt.plot(results['kinetic_energy'])
    plt.xlabel(f'Step,dt={dt}')
    plt.ylabel('Energy')
    plt.title('Energy Evolution')
    plt.legend(['Total Energy', 'Potential Energy', 'Kinetic Energy'])
    plt.grid(True)
    plt.savefig(f'energy_{temperature}.png')
    plt.clf()
    plt.close()
    # Plot temperature
    plt.figure(figsize=(12, 4))
    plt.plot(results['temperature'])
    plt.xlabel('Step')
    plt.ylabel('Temperature')
    plt.title('Temperature Evolution')
    plt.grid(True)
    plt.savefig(f'temperature_{temperature}.png')
    plt.clf()
    plt.close()

    # Plot RDF
    r, g_r = results['rdf']
    
    # positions = results['positions']
    plt.plot(r, g_r,label=f'g(r) Temperature= {temperature}')
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function (Last 2000 steps)')
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    plt.savefig(f'md_res_{temperature}.png')
    plt.clf()
    plt.close()
    
    # Plot temperature
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(results['temperature'])
    plt.xlabel('Step')
    plt.ylabel('Temperature')
    plt.title('Temperature Evolution')
    plt.show()