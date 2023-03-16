import os, operator
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


def find_checkpoint(rank, iteration=np.inf, operation='lt', reduction=max,
                        checkpoint_dir='./checkpoints'):
    """
    Look for and return the index of a valid checkpoint for the specified rank.
    
    Returns the `reduction` iteration for all checkpoints satisfying the 
    condition (`operation` `iteration`)

    Default: return the absolute highest iteration checkpoint.
    """
    # try to find checkpoints
    cdir = f'{checkpoint_dir}/proc{rank}/hdf5'
    if os.path.exists(cdir) and os.path.isdir(cdir):
        points = sorted([int(f[4:-3]) for f in os.listdir(cdir)])
        op = getattr(operator, operation)
        
        try:
            points = np.array(points)
            mask = op( points, iteration )
            restart_point = reduction( points[mask] )
                
            print(f'Selected checkpoint {restart_point} for rank {rank}')
            return restart_point
        except ValueError:
            print(f'Rank {rank} found no valid checkpoint for the parameters: reduction={reduction}, operation={op}, iteration={iteration})')
            return None
    else:
        print(f'Found no checkpoints for rank {rank}')
        return None
    


