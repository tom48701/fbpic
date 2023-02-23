import os
import numpy as np
from mpi4py import MPI


def remove_old_checkpoints( checkpoint_dir='checkpoints' ):
    """
    Automatically look for and remove all checkpoints within the specified
    directory except for the most recent (highest iteration) for each proc.
    No confirmation is given before deletion!
    """
    rank = MPI.COMM_WORLD.rank
    
    if rank == 0:
        try:
            procs = os.listdir(checkpoint_dir)
        except FileNotFoundError:
            print(f'Checkpoints not set, or are in a directory other than "{checkpoint_dir}"')
            return
        
        Nprocs = len(procs)
        
        for i in range(Nprocs):
            procdir = f'{checkpoint_dir}/proc{i}/hdf5'
            
            try:
                files = os.listdir(procdir) # files may not be sorted
            except FileNotFoundError:
                print(f'Checkpoints not set, or are in a directory other than "{checkpoint_dir}"')
                return
            points = [int(f[4:-3]) for f in files]
            imax = np.argmax(points)
            latest_file = files[imax]
            
            for f in files:
                if f != latest_file:
                    print(f'removing {checkpoint_dir}/proc{i}/hdf5/{f}')
                    os.remove(f'{checkpoint_dir}/proc{i}/hdf5/{f}')
                    
            print(f'leaving {checkpoint_dir}/proc{i}/hdf5/{latest_file}')
