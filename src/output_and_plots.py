import os
from datetime import datetime
from config import Configuration

def save_xyz(positions, filename, step):
    """
    Append particle positions to an XYZ file.
    Converts positions from meters to Angstroms for visualization.
    
    Args:
        positions (np.ndarray): Array of particle positions (N, 3) in meters.
        filename (str): Path to the XYZ file.
        step (int): Current simulation step number.
    """
    try:
        with open(filename, "a") as f:
            f.write(f"{len(positions)}\n")
            # Comment line can include step number
            f.write(f"Step: {step}\n") 
            for pos in positions:
                # Convert meters to Angstroms
                pos_A = pos * 1e10 
                f.write(f"Ar {pos_A[0]:12.6f} {pos_A[1]:12.6f} {pos_A[2]:12.6f}\n")
    except IOError as e:
        print(f"Error writing to xyz file {filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in save_xyz: {e}")