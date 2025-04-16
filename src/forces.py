from numba import jit
import numpy as np

@jit(nopython=True)
def compute_lj_force(r, sigma, epsilon, rcutoff):
    """Compute Lennard-Jones force magnitude with cutoff."""
    if r >= rcutoff or r < 1e-12:
        return 0.0
    inv_r = sigma / r
    inv_r6 = inv_r ** 6
    inv_r12 = inv_r6 ** 2
    # returns force magnitude in in N (J/m)
    return 24 * epsilon * (2 * inv_r12 - inv_r6) / r

@jit(nopython=True)
def compute_lj_potential(r, sigma, epsilon, rcutoff):
    """Compute shifted Lennard-Jones potential with zero at cutoff."""
    if r >= rcutoff:
        return 0.0
    inv_r = sigma / r
    inv_r6 = inv_r ** 6
    inv_r12 = inv_r6 ** 2
    potential = 4 * epsilon * (inv_r12 - inv_r6)

    # Shift potential to zero at cutoff
    inv_rcut = sigma / rcutoff
    inv_rcut6 = inv_rcut ** 6
    inv_rcut12 = inv_rcut6 ** 2
    shift = 4 * epsilon * (inv_rcut12 - inv_rcut6)
    # returns shifted potential in J
    return potential - shift

@jit(nopython=True)
def compute_forces_naive_virial(positions, box_size, rcutoff, sigma, epsilon):
    """
    Naive O(N^2) force calculation for 3D with PBC, including virial sum.
    
    Returns: forces (N), potential_energy (J), virial_sum (J, 6 components)
    """
    n = len(positions)
    forces = np.zeros_like(positions)   # N
    potential_energy = 0.0              # J
    virial_sum = np.zeros(6)            # [xx,yy,zz,xy,xz,yz] in J

    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]           # m
            # Apply minimum image convention
            rij -= box_size * np.round(rij / box_size)  # m

            r_sq = np.dot(rij, rij) # m2

            if r_sq < rcutoff**2:
                r = np.sqrt(r_sq)
                if r < 1e-12: continue # Avoid division by zero if particles overlap exactly

                f_mag = compute_lj_force(r, sigma, epsilon, rcutoff) # N
                f_vec = f_mag * (rij / r) # Force on i due to j (N)

                forces[i] += f_vec
                forces[j] -= f_vec # Newton's 3rd law

                potential_energy += compute_lj_potential(r, sigma, epsilon, rcutoff)

                # Pairwise Virial contribution: rij_alpha * Fij_beta (m * N = J)
                virial_sum[0] += rij[0] * f_vec[0] # W_xx
                virial_sum[1] += rij[1] * f_vec[1] # W_yy
                virial_sum[2] += rij[2] * f_vec[2] # W_zz
                virial_sum[3] += rij[0] * f_vec[1] # W_xy
                virial_sum[4] += rij[0] * f_vec[2] # W_xz
                virial_sum[5] += rij[1] * f_vec[2] # W_yz

    return forces, potential_energy, virial_sum

@jit(nopython=True)
def build_linked_cells(positions, box_size, rcutoff):
    """
    Assign particles to cells using the linked-cell algorithm.

    Parameters:
        positions : (N, 3) array of particle positions (m)
        box_size : float, size of the cubic simulation box (m)
        rcutoff : float, cutoff radius (m)

    Returns:
        head : list of head indices for each cell
        lscl : linked list of particle indices
        lc_dim : integer, number of cells in each dimension
        rc : actual cell size (scalar, m)
    """
    n_particles, dim = positions.shape
    dim = 3

    # Divide box into number of cells per dimension
    lc = max(1, int(np.floor(box_size / rcutoff)))
    rc = box_size / lc  # cell size (m)

    EMPTY = -1
    # Total nr of cells
    lc_yz = lc * lc
    lc_xyz = lc * lc_yz
    head = [EMPTY] * lc_xyz
    lscl = [EMPTY] * n_particles

    for i in range(n_particles):
        # Determine cell index vector (e.g., [x_idx, y_idx, z_idx]) based on particle's positions
        mc0 = min(lc - 1, int(positions[i, 0] / rc))
        mc1 = min(lc - 1, int(positions[i, 1] / rc))
        mc2 = min(lc - 1, int(positions[i, 2] / rc))

        # Convert 3D cell index vector (mc) to a scalar (1D) index for accessing the 'head' array
        c_index = mc0 * lc_yz + mc1 * lc + mc2

        # Add the particle to the linked list of particles in the appropriate cell
        lscl[i] = head[c_index]
        head[c_index] = i

    return head, lscl, lc, rc

@jit(nopython=True)
def compute_forces_lca_virial(positions, box_size, rcutoff, sigma, epsilon):
    """
    Compute LJ forces, potential energy, and virial sum using LCA (3D, PBC).
    
    Returns:
        forces (N), potential_energy (J), virial_sum (J, 6 components)
    """
    n_particles = positions.shape[0]
    dim = 3
    # Build the linked cells
    head, lscl, lc, rc = build_linked_cells(positions, box_size, rcutoff)

    forces = np.zeros_like(positions)
    potential_energy = 0.0
    virial_sum = np.zeros(6)  # [xx,yy,zz,xy,xz,yz] in J
    EMPTY = -1

    # Create offsets for 27 neighboring cells (including self), Numba-compatible
    neighbor_offsets = np.array([
        [dx, dy, dz]
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ])

    lc_yz = lc * lc  # Precompute for index calculation

    # Iterate over cells
    for mc0 in range(lc):
        for mc1 in range(lc):
            for mc2 in range(lc):
                # Scalar index of the origin cell
                c_index_origin = mc0 * lc_yz + mc1 * lc + mc2
                # Get the first particle 'i' in the origin cell
                i = head[c_index_origin]

                while i != EMPTY:
                    pos_i = positions[i]  # Position of particle i (m)
                    # Iterate over neighboring cells (including the origin cell itself)
                    for offset in neighbor_offsets:
                        # Index vector of the neighbor cell
                        mc_neighbor = np.array([mc0, mc1, mc2]) + offset
                        rshift = np.zeros(dim)  # Shift due to PBC wrap-around

                        # Apply PBC to neighbor cell indices and determine shift
                        for a in range(dim):
                            if mc_neighbor[a] < 0:
                                mc_neighbor[a] += lc
                                rshift[a] = -box_size
                            elif mc_neighbor[a] >= lc:
                                mc_neighbor[a] -= lc
                                rshift[a] = box_size

                        # Scalar index of the neighbor cell
                        c_index_neighbor = (
                            mc_neighbor[0] * lc_yz + mc_neighbor[1] * lc + mc_neighbor[2]
                        )

                        # Get the first particle 'j' in the neighbor cell
                        j = head[c_index_neighbor]
                        while j != EMPTY:
                            if c_index_origin < c_index_neighbor or (
                                c_index_origin == c_index_neighbor and j > i
                            ):
                                # Position of particle j, adjusted by PBC shift (m)
                                pos_j_shifted = positions[j] + rshift

                                # Vector from j to i (m)
                                rij = pos_i - pos_j_shifted
                                r_sq = np.dot(rij, rij)  # m^2

                                if r_sq < rcutoff ** 2:
                                    r = np.sqrt(r_sq)  # m
                                    if r < 1e-12:
                                        j = lscl[j]
                                        continue  # Avoid division by zero

                                    # Compute force magnitude and vector (force on i due to j)
                                    f_mag = compute_lj_force(r, sigma, epsilon, rcutoff)  # N
                                    f_vec = f_mag * (rij / r)  # N

                                    # Accumulate forces
                                    forces[i] += f_vec
                                    forces[j] -= f_vec  # Newton's 3rd law

                                    # Accumulate potential energy
                                    potential_energy += compute_lj_potential(
                                        r, sigma, epsilon, rcutoff
                                    )  # J

                                    # Accumulate Pairwise Virial contribution: rij_alpha * Fij_beta (J)
                                    virial_sum[0] += rij[0] * f_vec[0]  # W_xx
                                    virial_sum[1] += rij[1] * f_vec[1]  # W_yy
                                    virial_sum[2] += rij[2] * f_vec[2]  # W_zz
                                    virial_sum[3] += rij[0] * f_vec[1]  # W_xy
                                    virial_sum[4] += rij[0] * f_vec[2]  # W_xz
                                    virial_sum[5] += rij[1] * f_vec[2]  # W_yz

                            # Move to the next particle 'j' in the neighbor cell's list
                            j = lscl[j]
                    # Move to the next particle 'i' in the origin cell's list
                    i = lscl[i]

    return forces, potential_energy, virial_sum