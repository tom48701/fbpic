import subprocess, os
from mpi4py import MPI

# fitted coefficients 
ptclcoeff = 0.0001380681991557795
gridcoeffs = [ 1.30049848e-04,  2.95065904e-04,  1.04118576e+01, -1.17777077e-08,
               1.09971430e-01,  8.43420351e-02,  6.68461583e-04,  8.11783373e+02]

def ptcl(Np, m=ptclcoeff):
    """ fitting function for particle memory use """
    return Np*m

def grid(x,m1,m2,m3,m12,m23,m13,m123,c):
    """ fitting function for grid memory use """
    return c + m1*x[0] + m2*x[1] + m3*x[2] + m12*x[0]*x[1] + m23*x[1]*x[2] + \
        m13*x[0]*x[2] + m123*x[0]*x[1]*x[2]

def estimate_memory(Nz, Nr, Nm, Np, gridcoeffs=gridcoeffs, ptclcoeff=ptclcoeff):
    """ Estimate the total memory needed by a simulation """
    gridmem = grid([Nz,Nr,Nm], *gridcoeffs)
    ptclmem = ptcl(Np, ptclcoeff)
    
    mem = gridmem + ptclmem 
    print(f'Memory estimates for grid size {Nz:d}x{Nr:d} using {Nm:d} modes\ncontaining {Np/1e6:.2f}M particles')
    print(f'  Grid {gridmem:.0f} MiB\n  Particles {ptclmem:.0f} MiB')
    return mem
    
    
def memuse(sim=None, write=True, gridcoeffs=gridcoeffs, ptclcoeff=ptclcoeff):
    """
    report GPU memory use for the matching PID
    if a `sim` is provided, pull grid and particle data out and try to
    estimate the individual contributions using empirically-determined
    scaling factors `fldcoeff' and `ptclcoeff`.
    """
    rank = MPI.COMM_WORLD.rank
    mem = 'N/A'
    # get PID
    ospid = os.getpid()
    rep = f'Rank {rank} PID: {ospid}\n'
    # get GPU stats, clean the output - could do this with grep etc., but then would break on windows
    smi = subprocess.check_output('nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader',
                                  shell=True)
    smi = str(smi, 'utf-8')
    smi = smi.replace('\r','')
    smi = smi.split('\n')
    for line in smi[:-1]:
        split = line.split(', ')
        pid = int(split[1])
        if pid == ospid:
            uuid = split[0]
            mem = split[2]
            rep += f'  {uuid}: {mem}\n'
            break

    if sim is not None and write:
        # grid stats
        Nz = sim.fld.Nz
        Nr = sim.fld.Nr
        Nm = sim.fld.Nm
        gridmem = grid([Nz,Nr,Nm], *gridcoeffs)
        rep += f'Roughly {gridmem:.2f} MiB allocated for the grid\n'
        # particle stats
        total_particles = 0
        for i in range(len(sim.ptcl)):
            spec = sim.ptcl[i]
            Ntot = spec.Ntot
            total_particles += Ntot
            rep += f'  species {i}: {Ntot/1e6:.2f}M particles / roughly {ptcl(Ntot):.0f} MiB\n'
            
        rep += f'Roughly {ptcl(total_particles):.0f} MiB allocated for the particles\n'
    
    if write:
        print(rep)
    return mem
