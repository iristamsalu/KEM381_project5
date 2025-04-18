import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def read_last_xyz_frames(filename, num_frames=1):
    """
    Read the last N frames from an XYZ file.
    
    Args:
        filename (str): Path to XYZ file
        num_frames (int): Number of frames to read from the end
        
    Returns:
        list: List of numpy arrays containing particle positions for each frame
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    frames = []
    current_frame = []
    atom_count = 0
    lines_read = 0
    
    # Process lines in reverse to efficiently get last frames
    for line in reversed(lines):
        parts = line.strip().split()
        
        if len(parts) == 1 and parts[0].isdigit():
            # This is an atom count line (start of frame)
            atom_count = int(parts[0])
            if current_frame:
                frames.append(np.array(current_frame))
                if len(frames) >= num_frames:
                    break
                current_frame = []
        elif len(parts) >= 4:  # Atom line (element, x, y, z)
            current_frame.insert(0, [float(parts[1]), float(parts[2]), float(parts[3])])
    
    # Handle the first frame if we didn't complete it
    if current_frame and len(frames) < num_frames:
        frames.append(np.array(current_frame))
    
    return frames[-num_frames:]  # Return only the requested number of frames

def compute_rdf(positions, density, r_cutoff, n_bins, num_particles):
    delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.sqrt(np.sum(delta**2, axis=-1))
    
    # Bin and normalize RDF
    hist, bin_edges = np.histogram(distances[distances > 0], bins=n_bins, range=(0, r_cutoff))
    r_values = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    dr = r_cutoff / n_bins
    shell_volumes = 4 * np.pi * r_values**2 * dr
    g_r = hist / (shell_volumes * density * num_particles)
    
    return r_values, g_r / 2  # Account for double-counting

def average_rdf_from_xyz(xyz_file, density, r_cutoff, n_bins, num_frames, num_particles):
    frames = read_last_xyz_frames(xyz_file, num_frames)
    
    if not frames:
        raise ValueError("No valid frames found in XYZ file")
    
    # Initialize RDF accumulation
    r_values = None
    total_g_r = np.zeros(n_bins)
    
    for frame in frames:
        current_r, current_g_r = compute_rdf(frame, density, r_cutoff, n_bins,num_particles)
        if r_values is None:
            r_values = current_r
        total_g_r += current_g_r
    
    avg_g_r = total_g_r / len(frames)
    
    return r_values, avg_g_r

# Example usage:
if __name__ == "__main__":
    sigma = 3.405e-10
    density = 0.0322
    num_particles = 256
    num_frames = 1000
    xyz_file= 'trajectory.xyz'
    n_bins=500
    r, g_r = average_rdf_from_xyz(xyz_file=xyz_file, 
                                 density=density, 
                                 r_cutoff=5*sigma,
                                 n_bins=n_bins, 
                                 num_frames=num_frames,
                                 num_particles=num_particles)
    plt.figure(figsize=(15, 12))
    plt.grid(True)
    plt.plot(r/sigma, g_r)
    plt.xlabel("Distance (r/sigma)")
    plt.ylabel("g(r)")
    plt.title(f"Average RDF from last {num_frames} frames")
    plt.show()