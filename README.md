# KEM381
## Project Assignment 5

### Overview
This is a simple Lennard-Jones (LJ) molecular dynamics (MD) model for Argon. The simulation can be run as NVE, NVT or NPT ensemble. NB! NPT is not properly tested. In addition, the user can visualize phase behaviour with radial distribution function (RDF) or calculate viscosity with Green-Kubo method.

### Files
- **main.py**: Main execution script that parses arguments, initializes the simulation, and runs the LJ simulation.
- **simulation.py**: Core of the simulation logic. Manages system initialization, lattice setup, velocity initialization, force and virial calculations, pressure tensor computation, energy tracking, Velocity Verlet integration, thermostat and barostat control, periodic boundary conditions, energy minimization, and XYZ trajectory/output file saving.
- **thermostats.py**: Contains three thermostat functions: Langevin, Berendsen and Nose-Hoover, and Parrinello-Rahman barostat.
- **forces.py**: Implements Lennard-Jones force and potential calculations using linked cell algorithm (LCA).
- **config.py**: Parses and validates command-line arguments, defines Lennard-Jones parameters for argon (σ, ε, mass), and stores all simulation settings in a structured Configuration dataclass.
- **rdf.py**: Computes and plots RDF.
- **green_kubo.py**: Calculates viscosity with Green-Kubo method.
- **requirements.txt**: A list of packages and libraries needed to run the programs.


### Installing Requirements
To install the necessary requirements in a virtual environment, use the following command:
```
pip3 install -r requirements.txt
```

### Running the Main MD Program
To run the Argon MD simulation program, you need to provide certain parameters through the command line.

#### Explanation of Arguments:

- `--steps <number_of_steps>`: Number of simulation steps (default: 20000).
- `--dt <time_step>`: Time step size in seconds (default: 2e-15 s).
- `--n_particles <number_of_particles>`: Number of particles in the simulation box (default: 1024).
- `--temperature <temperature>`: Simulation temperature in Kelvin (default: 173 K).
- `--density <density>`: Mass density in kg/m³ (default: 844.53 kg/m³).
- `--sigma <LJ_sigma>`: Lennard-Jones sigma parameter (default: 3.40e-10 m, for Argon).
- `--epsilon <LJ_epsilon>`: Lennard-Jones epsilon parameter (default: 1.65678e-21 J, for Argon).
- `--mass` <particle_mass>: Particle mass in kilograms (default: 6.6335e-26 kg, for Argon).
- `--rcutoff <LJ_cutoff_radius>`: Cutoff radius for Lennard-Jones potential (default: 2.5 × sigma).
- `--use_langevin`: (Optional) Apply Langevin thermostat for temperature control (mutually exclusive with --use_berendsen).
- `--use_berendsen`: (Optional) Apply Berendsen thermostat for temperature control (mutually exclusive with --use_langevin).
- `--thermostat_constant`: Thermostat parameter (friction coefficient for Langevin, relaxation time for Berendsen). If not specified, a default is calculated based on particle mass.
- `--npt`: (Optional) Run simulation in NPT ensemble (requires pressure coupling).
- `--nvt`: (Optional) Run simulation in NVT ensemble (temperature controlled, volume fixed).
- `--nve`: (Optional) Run simulation in NVE ensemble (no thermostat/barostat; energy conserving).
- `--volume` <initial_volume>: (Optional) Manually specify box volume (in m³); otherwise computed from density.
- `--init_config` <file.xyz>: (Optional) Start from an existing XYZ file with particle coordinates.
- `--target_pressure` <pressure>: Target pressure for NPT ensemble, in Pascals (default: 16.1789 MPa).
- `--nh_Q` <Nose_Hoover_Q>: Coupling constant for Nose-Hoover barostat (default: 4e-4).
- `--pr_W` <Parrinello_Rahman_W>: Coupling constant for Parrinello-Rahman barostat (default: 1e-20).


#### Example Commands:
- **Example 1: NVE**:
    ```
    python main.py --nve --temperature 223 --density 3.51

    ```

- **Example 2: NVT**:
    ```
    python main.py --nvt --temperature 223 --use_langevin --density 3.51

    ```

- **Example 3: NPT**:
    ```
    python main.py --npt --temperature 223 --density 3.51 --target_pressure 0.1628e6

    ```

- **Example 4: NPT-NVT-NVE**:
    ```
    python main.py --npt --temperature 223 --density 3.51 --target_pressure 0.1628e6
    python main.py --nvt --temperature 223 --use_langevin --density 3.51 --init_config ./output/npt_final.xyz --volume 1.935487e-23
    python main.py --nve --temperature 223 --density 3.51 --init_config ./output/nvt_final.xyz --volume 1.935487e-23
    ```

#### MD simulation output:
1. Program creates output folder and several output files.
2. Trajectory is saved to `trajectory.xyz` file which can be visualized using tools like VMD or Ovito.
3. Detailed energy values during the simulation are stored in `energy.dat`. It also tracks temperature and volume.
4. Pressure component (Pxy, Pxz, Pyz) for green_kubo.py are saved to `pressure.dat`.
5. NPT and NVT final frames are saved to `npt_final.xyz` and `nvt_final.xyz`.

---

### Running rdf.py
To calculate RDF you need to have a `.XYZ` file of a trajectory.

#### Explanation of Arguments:

- `<filename>`: (required) XYZ file that you want to use or path to the input XYZ file
- `--density <density>`: (required) Mass density in kg/m³
- `--start <starting frame for RDF calculations>`: (optional) Give a starting frame if you want to skip some of the initial frames (default: 0)
- `--bins <nr of histogram bins>`: (optional) Number of RDF histogram bins for resolution (default: 100)

#### Example Command:
- **Example: Compute RDF for a simulation trajectory**:
    ```
    python rdf.py ./output/trajectory.xyz --density 844.53 --start 100 --bins 100

    ```

#### rdf.py output:
1. Program creates a rdf_plot.png file into the output folder.

---

### Running green_kubo.py
To calculate viscosity you need to have `pressure.dat` file. NB! green_kubo.py does not accept command-line arguments; you must edit variable values directly in the script.

#### Explanation of Important Arguments:
- `DATA_FILE`: `./output/pressure.dat` or another file with similar content
- `VOLUME`: box volume in m^3
- `TEMPERATURE`: temperature in Kelvins
- `INTEGRATION_START_PS` - adjust to change the integration interval
- `INTEGRATION_END_PS` - adjust to change the integration interval

#### Example Command:
- **Example: After adjusting the variables in green_kubo.py**:
    ```
    python green_kubo.py

    ```

#### green_kubo.py output:
1. Program creates a viscosity_plot.png file into the output folder.