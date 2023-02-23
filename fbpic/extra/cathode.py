import warnings, os, operator
import numpy as np
from scipy.constants import c

from fbpic.fields import Fields
from fbpic.particles import Particles
from fbpic.lpa_utils.bunch import get_space_charge_spect
from fbpic.particles.elementary_process.cuda_numba_utils import \
    reallocate_and_copy_old

from mpi4py import MPI

class Cathode(Particles):
    def __init__(self, sim, q, m, z_cathode,
                 x,y,z,ux,uy,uz,w,
                 initialize_self_field=True, 
                 direction='forward', rmax=None,
                 t_min=-np.inf, t_max=np.inf,
                 checkpoint_dir='./checkpoints',
                 checkpoint_period=None,
                 checkpoint_all_beam_info=True):

        Particles.__init__(self, q=q, m=m, n=0., Npz=0, zmin=0., zmax=0., Npr=0, 
                           rmin=0., rmax=0., Nptheta=0, dt=sim.dt,
                    dens_func=None, continuous_injection=False,
                    grid_shape=sim.grid_shape, particle_shape=sim.particle_shape,
                    use_cuda=sim.use_cuda, dz_particles=0.)
        # warning
        warnings.warn('This cathode will do nothing unless you are using a modified copy of fbpic.main!')
        # sanity checks
        assert direction in ('forward', 'backward')
        # internal attribute used to trigger injection inside the main loop
        # there is possibly a better way to do this?
        self._is_cathode = True
        
        # Register the arguments
        self.sim = sim
        self.z_cathode = z_cathode
        self.direction = direction
        self.q = q
        self.m = m
        self.t_min = t_min
        self.t_max = t_max
        self.rmax = rmax
        self.initialize_self_field = initialize_self_field
        
        # initialise the injector object for the cathode
        self.injector = CathodeContinuousInjector( sim, z_cathode, x,y,z,ux,uy,uz,w, 
                                                  direction=direction)

        self.checkpoint_period = checkpoint_period
        self.checkpoint_dir = checkpoint_dir
        # to restart, the injection mask must be saved, but in the case of 
        # randomly generated bunches, the whole beam must be saved as well
        # in order to preserve continuity. However, this is not required if 
        # loading from an array, so make it an option
        self.checkpoint_all_beam_info = checkpoint_all_beam_info
        
        # prep for checkpoints 
        if checkpoint_period is not None:
            comm = MPI.COMM_WORLD
            if comm.rank == 0:
                if os.path.exists(checkpoint_dir) is False:
                    os.mkdir(checkpoint_dir)            
            comm.barrier()
            
        return
    
    def inject_particles( self, time): 
        """ 
        Inject particles per subdomain. 
        """
        # check the time
        if time < self.t_min or time > self.t_max:
            return

        # Have the injector look for new particles
        Ntot, x, y, z, ux, uy, uz, inv_gamma, w = \
                            self.injector.generate_particles( time )
                        
        # clip particles at rmax if required
        if self.rmax is not None:
            r = np.sqrt(x**2+y**2)
            
            mask = r < self.rmax
            Ntot = mask.sum()
            
            x = x[mask]
            y = y[mask]
            z = z[mask]
            ux = ux[mask]
            uy = uy[mask]
            uz = uz[mask]
            inv_gamma = inv_gamma[mask]
            w = w[mask]   

        if Ntot > 0:
            print( f'{self.sim.comm.rank}: {Ntot}')
            # get the existing number of particles
            old_Ntot = self.Ntot
            # expand the particle arrays to accomodate new particles
            Ntot = old_Ntot + Ntot
            reallocate_and_copy_old( self, self.use_cuda, old_Ntot, Ntot )

            if self.use_cuda:
                self.receive_particles_from_gpu()
            # fill the resized buffers with new particle data
            self.x[old_Ntot:] = x[:]
            self.y[old_Ntot:] = y[:]
            self.z[old_Ntot:] = z[:]
            self.ux[old_Ntot:] = ux[:]
            self.uy[old_Ntot:] = uy[:]
            self.uz[old_Ntot:] = uz[:]
            self.inv_gamma[old_Ntot:] = inv_gamma[:]
            self.w[old_Ntot:] = w[:]
            # create new IDs
            if self.tracker is not None:
                new_ids = self.tracker.generate_new_ids( Ntot - old_Ntot )            
                self.tracker.id[old_Ntot:] = new_ids[:]
                
            # particles must then be sorted on GPU
            if self.use_cuda:
                self.sorted = False
                self.send_particles_to_gpu()
        # add the self fields if required
        if self.initialize_self_field:
            self.add_self_field( x,y,z,ux,uy,uz,w,inv_gamma )
    
    def restart_from_checkpoint(self, iteration=None):
        """ 
        Set the beam injection state from a checkpoint.
        Choose the latest (highest numbered) checkpoint if none is specified
        """
        index = self.sim.ptcl.index(self)
        
        files = os.listdir( self.checkpoint_dir )
        valid_files = [f.startswith(f'cathode{index}') and f.endswith('.npz') for f in files]
        # select the valid files
        files = [ f[-12:-4] for f in np.array(files)[valid_files] ]
        iterations = [int(i) for i in files]
            
        if iteration is None:
            # choose the latest checkpoint if none specified
            iteration = max(iterations)

        # match behaviour to the simulation restart function
        i_iteration = ( abs(np.array(iterations) - iteration) ).argmin()
        iteration = iterations[i_iteration]
    
        fname = os.path.join(self.checkpoint_dir, f'cathode{index}_{iteration:08d}.npz')
        
        with np.load( fname,'r') as new_beam:
            
            self.injector.overwrite_beam( new_beam )
        
        return
             
    def add_self_field( self, x,y,z,ux,uy,uz,w,inv_gamma ):
        """
        Add the self fields from the given particles to the simulation
        """
        # shortcuts
        sim = self.sim

        # Create an empty particle species
        ptcl = Particles(q=self.q, m=self.m, n=0., Npz=0, zmin=0., zmax=0., Npr=0, 
                               rmin=0., rmax=0., Nptheta=0, dt=sim.dt,
                        dens_func=None, continuous_injection=False,
                        grid_shape=sim.grid_shape, particle_shape=sim.particle_shape,
                        use_cuda=sim.use_cuda, dz_particles=0.)
        
        # Reallocate buffer arrays with the right number of electrons
        Ntot = len(x)  
        reallocate_and_copy_old( ptcl, ptcl.use_cuda, 0, Ntot )

        # Fill the resized particle arrays with the right values
        ptcl.x[:] = x[:]
        ptcl.y[:] = y[:]
        ptcl.z[:] = z[:]
        ptcl.ux[:] = ux[:]
        ptcl.uy[:] = uy[:]
        ptcl.uz[:] = uz[:]
        ptcl.inv_gamma[:] = inv_gamma[:]
        ptcl.w[:] = w[:]

        # Calculate the mean gamma by computing weighted sum on each subdomain
        w_sum_local = ptcl.w.sum()
        w_gamma_sum_local = (ptcl.w*1./ptcl.inv_gamma).sum()
        if sim.comm.mpi_comm is None:
            w_sum = w_sum_local
            w_gamma_sum = w_gamma_sum_local
        else:
            w_sum = sim.comm.mpi_comm.allreduce(w_sum_local)
            w_gamma_sum = sim.comm.mpi_comm.allreduce(w_gamma_sum_local)
        # Check that the number of particles is not 0
        if w_sum == 0:
            return
        else:
            gamma = w_gamma_sum/w_sum

        # The particles must be on GPU for the depositions, but this species 
        # will not be handled by the GpuMemoryManager as it is not in sim.ptcl,
        # so we move the particles manually
        if sim.use_cuda: 
            ptcl.send_particles_to_gpu()
            
        # Project the charge and currents onto the local subdomain
        sim.deposit( 'rho', exchange=True, species_list=[ptcl],
                        update_spectral=False )
        sim.deposit( 'J', exchange=True, species_list=[ptcl],
                        update_spectral=False )

        # Simulation fields must be on CPU to be gathered for the global field
        # Return the particles as well
        if sim.use_cuda:
            sim.fld.receive_fields_from_gpu() 
            
        # Create a global field object across all subdomains
        # (Space-charge calculation is a global operation)
        # Note: in the single-proc case, this is also useful in order not to
        # erase the pre-existing E and B field in sim.fld
        global_Nz, _ = sim.comm.get_Nz_and_iz(
                        local=False, with_damp=True, with_guard=False )
        global_zmin, global_zmax = sim.comm.get_zmin_zmax(
                        local=False, with_damp=True, with_guard=False )
        global_fld = Fields( global_Nz, global_zmax,
                sim.fld.Nr, sim.fld.rmax, sim.fld.Nm, sim.fld.dt,
                n_order=sim.fld.n_order, smoother=sim.fld.smoother,
                zmin=global_zmin, use_cuda=False)
        
        # Gather the sources on the interpolation grid of global_fld
        for m in range(sim.fld.Nm):
            for field in ['Jr', 'Jt', 'Jz', 'rho']:
                local_array = getattr( sim.fld.interp[m], field )
                gathered_array = sim.comm.gather_grid_array(
                                                local_array, with_damp=True )
                setattr( global_fld.interp[m], field, gathered_array )
                
        # Calculate the space-charge fields on the global grid
        # (For a multi-proc simulation: only performed by the first proc)
        if sim.comm.rank == 0:
            # - Convert the sources to spectral space
            global_fld.interp2spect('rho_prev')
            global_fld.interp2spect('J')
            if sim.filter_currents:
                global_fld.filter_spect('rho_prev')
                global_fld.filter_spect('J')
            # - Get the space charge fields in spectral space
            for m in range(global_fld.Nm) :
                get_space_charge_spect( global_fld.spect[m], gamma, self.direction )
            # - Convert the fields back to real space
            global_fld.spect2interp( 'E' )
            global_fld.spect2interp( 'B' )

        # Communicate the results from proc 0 to the other procs
        # and add it to the interpolation grid of sim.fld.
        # - First find the indices at which the fields should be added
        Nz_local, iz_start_local_domain = sim.comm.get_Nz_and_iz(
            local=True, with_damp=True, with_guard=False, rank=sim.comm.rank )
        _, iz_start_local_array = sim.comm.get_Nz_and_iz(
            local=True, with_damp=True, with_guard=True, rank=sim.comm.rank )
        iz_in_array = iz_start_local_domain - iz_start_local_array
        # - Then loop over modes and fields
        for m in range(sim.fld.Nm):
            for field in ['Er', 'Et', 'Ez', 'Br', 'Bt', 'Bz']:
                # Get the local result from proc 0
                global_array = getattr( global_fld.interp[m], field )
                local_array = sim.comm.scatter_grid_array(
                                        global_array, with_damp=True )
                # Add it to the fields of sim.fld
                local_field = getattr( sim.fld.interp[m], field )
                local_field[ iz_in_array:iz_in_array+Nz_local, : ] += local_array

        # Return fields to the GPU
        if sim.use_cuda:
            sim.fld.send_fields_to_gpu()

        # Since we are doing this from within the PIC loop, we must update the 
        # spectral grid. This step is not usually neccesary when initialising 
        # bunches as the spectral grid is automatically updated when `step`
        # is called.
        sim.fld.interp2spect('E')
        sim.fld.interp2spect('B')
        return

    def write(self, iteration ):
        """
        Use this method to piggyback the diagnostics and write checkpoints
        """
        if self.checkpoint_period is not None:
            if iteration % self.checkpoint_period == 0:
                
                index = self.sim.ptcl.index(self)
                fname = f'cathode{index}_{iteration:08d}'
                self.injector.communicate_injection_mask()
                if self.sim.comm.rank == 0:
                    
                    inj = self.injector
                    
                    if self.checkpoint_all_beam_info:
                        np.savez(os.path.join(self.checkpoint_dir, fname), 
                                x=inj.x, y=inj.y, z=inj.z,
                                ux=inj.ux, uy=inj.uy, uz=inj.uz, w=inj.w,
                                injection_mask=inj.injection_mask )
                    else:
                        np.savez(os.path.join(self.checkpoint_dir, fname), 
                                injection_mask=inj.injection_mask )



class CathodeContinuousInjector:
    """
    Class that selects which particles are to be injected and communicates
    between procs
    """

    def __init__(self, sim, z_cathode, x,y,z,ux,uy,uz,w, 
                 direction='forward'):
        """
        register and sort the particle information, in order for it to be used
        to select particles to inject.

        Parameters
        ----------
        sim : Simulation object
            Parent simulation
            
        z_cathode : float
            The position of the cathode. Particles will be considered valid
            for injection if they are > this position (forward) or < this 
            position (backward)
    
        x, y, z, ux, uy, uz, w : arrays of floats
            All the particle information for the beam to be injected. 
            This is stored and used as reference.
    
        direction : string, optional
            Can be either "forward" or "backward".
            Propagation direction of the beam.
        
        """
        self.sim = sim
        self.size = sim.comm.size
        self.rank = sim.comm.rank
        # Register the cathode position
        self.z_cathode = z_cathode
        # Register the particle flow direction 
        self.direction = direction
        # choose a comparator for finding which particles to inject
        if direction == 'forward':
            self.compare = operator.gt
        elif direction == 'backward':
            self.compare = operator.lt
            
        # sort all the particles 
        sort = np.argsort(z)
        self.x = x[sort]
        self.y = y[sort]
        self.z = z[sort]
        self.ux = ux[sort]
        self.uy = uy[sort]
        self.uz = uz[sort]
        self.w = w[sort]
        self.inv_gamma = 1./np.sqrt(1.+ux**2+uy**2+uz**2)
        # initialise the injection mask
        self.injection_mask = np.full(len(x), True, dtype=bool)
        # auxilliary MPI array for reduction
        if self.size > 1:
            self.reduced_mask = np.full(len(x), True, dtype=bool)
        return
    
    def communicate_injection_mask(self):
        """ 
        The injection mask must be shared between procs to ensure that 
        particles are only injected once, perform a reduction and a broadcast
        to accomplish this
        """
        # Without modifying the fbpic communicator, the recude/bcast occurs 
        # over the whole MPI.COMM_WORLD, thus we communicate only if 
        # sim.comm.size > 1. i.e. we do not initiate communication for
        # parallel scans (use_all_mpi_ranks == False)
        # AFAIK you cannot run parallel scans with more than one proc per scan
        # however if this is _not_ the case, then this method is not robust!!
        if self.size > 1:
            comm = self.sim.comm.mpi_comm
            # reduce the masks onto rank 0 using the logical AND operator 
            comm.Reduce( self.injection_mask, self.reduced_mask, op=MPI.LAND, root=0)
            # rebroadcast the result
            comm.Bcast( self.reduced_mask, root=0)
            self.injection_mask[:] = self.reduced_mask[:]
        return
    
    def overwrite_beam(self, new_beam):
        """ 
        Overwrite the current beam using the contents of a npz file.
        The injection state is required, the full beam is optional
        """
        self.injection_mask[:] = new_beam['injection_mask'][:]
        # try to overwrite the rest of the beam, silently pass if the data is not present
        try:
            self.x[:] = new_beam['x'][:]
            self.y[:] = new_beam['y'][:]
            self.z[:] = new_beam['z'][:]
            self.ux[:] = new_beam['ux'][:]
            self.uy[:] = new_beam['uy'][:]
            self.uz[:] = new_beam['uz'][:]
            self.w[:] = new_beam['w'][:]
        except KeyError:
            pass
        return
        
    def generate_particles(self, time ):
        """ find the particles to be injected on each rank """
        
        # move the particles to their 'current' position
        x = self.x + time * self.ux * self.inv_gamma * c
        y = self.y + time * self.uy * self.inv_gamma * c
        z = self.z + time * self.uz * self.inv_gamma * c

        # build the mask for the particles
        # eliminate particles outside the local subdomain first
        zmin, zmax = self.sim.comm.get_zmin_zmax(
            local=True, with_damp=False, with_guard=False, rank=self.rank )
        mask = (z >= zmin) & (z < zmax)
        # next find particles eligible for injection based on position
        mask = mask & self.compare(z, self.z_cathode)
        # lastly mask off any already injected particles
        mask = mask & self.injection_mask
        # update the injection mask, then synchronise with the other ranks
        self.injection_mask = self.injection_mask & ~mask
        if self.size > 1:
            self.communicate_injection_mask()
    
        # mask the arrays, return the particles to be injected
        if mask.sum() > 0:
            x = x[mask]
            y = y[mask]
            z = z[mask]
            ux = self.ux[mask]
            uy = self.uy[mask]
            uz = self.uz[mask]
            w = self.w[mask]
            inv_gamma = self.inv_gamma[mask]
            Ntot = len(x)
            return( Ntot, x, y, z, ux, uy, uz, inv_gamma, w )
        else:  
            return( 0, np.zeros(0),np.zeros(0),np.zeros(0),np.zeros(0),
                np.zeros(0),np.zeros(0),np.zeros(0),np.zeros(0))

def add_gaussian_bunch_cathode( sim, q, m, sig_r, sig_z, n_emit, 
                                    gamma0, sig_gamma, 
                                    n_physical_particles, n_macroparticles,
                                    z_cathode, zf=0., tf=0., 
                                    t_min=-np.inf, t_max=np.inf, rmax=None, 
                                    initialize_self_field=True,
                                    direction='forward', save_beam=None,
                                    checkpoint_dir='./checkpoints',
                                    checkpoint_period=None,
                                    checkpoint_all_beam_info=True):
    """
    Initialise a Gaussian particle bunch with the given parameters and 
    incrementally introduce the bunch to the simulation as it passes the
    specified injection plane.

    Parameters
    ----------
    See `lpa_utils.bunch.add_particle_bunch_gaussian` for most options.
    
    z_cathode: float
        Position of the cathode. 
        Particles positions are initialised based on `zf` and `tf` and 
        propagate balistically off-grid as the simulation progresses. They are 
        then added to the simulation when their z coordinate is larger 
        (direction == 'forward') or smaller (direction == backward') than
        this value.

    t_min, t_max: floats, optional
        Perform no injection before or after these times respectively.
        Default: -inf, inf
        
    rmax: float, optional
        Clip the beam at this radius, useful to prevent particles being 
        initialised outside the box.
        Default: None
        
    checkpoint_dir: str, optional
        Directory to save the checkpoint information in
        Default : `./checkpoints`
        
    checkpoint_all_beam_info: bool, optional
        If `True`, save all beam information to file each time.
        If `False`, save only the injection mask.
        Default: True
        
    Returns
    -------
    cathode: Cathode object
        The newly-created cathode, behaves as a Particles object in most cases
        
    Notes
    -----
    If the beam is initialised with some randomised elements, the exact
    position and momentum distribution of the beam must be kept consistent
    between restarts. 
    This is easiest done by leaving 'checkpoint_all_beam_info' as `True`, in 
    which case the full beam information is saved in each dump, and will 
    automatically be used to regenerate the beam upon restart.
    If the beam is very large, or restarts very frequent, it may be more 
    economical to initialise the beam from a pre-saved file each time, via
    `add_cathode_from_arrays`, in which case only the injection mask is 
    needed to preserve continuity.
    
    Adding the particle self-fields is a very expensive operation, 
    it depends on the global grid size and is performed on CPU, in serial.
    This is not much of a concern when called once per simulation for a 
    directly initialised bunch, but cathodes are continually injecting 
    particles.
    It is likely this operation will dominate execution time for the duration
    of the injection, and if the grid is very large, it can significantly 
    impact performance. 
    """
    # build the whole particle beam (on first proc only!) (from bunch.py)
    if sim.comm.rank == 0:
        # Generate Gaussian gamma distribution of the beam
        if sig_gamma > 0.:
            gamma = np.random.normal(gamma0, sig_gamma, n_macroparticles)
        else:
            # Zero energy spread beam
            gamma = np.full(n_macroparticles, gamma0)
            if sig_gamma < 0.:
                warnings.warn(
                    "Negative energy spread sig_gamma detected."
                    " sig_gamma will be set to zero. \n")
        # Get inverse gamma
        inv_gamma = 1. / gamma
        # Get Gaussian particle distribution in x,y,z
        x = sig_r * np.random.normal(0., 1., n_macroparticles)
        y = sig_r * np.random.normal(0., 1., n_macroparticles)
        z = zf + sig_z * np.random.normal(0., 1., n_macroparticles)
    
        # Define sigma of ux and uy based on normalized emittance
        sig_ur = (n_emit / sig_r)
        # Get Gaussian distribution of transverse normalized momenta ux, uy
        ux = sig_ur * np.random.normal(0., 1., n_macroparticles)
        uy = sig_ur * np.random.normal(0., 1., n_macroparticles)
    
        # Finally we calculate the uz of each particle
        # from the gamma and the transverse momenta ux, uy
        uz_sqr = (gamma ** 2 - 1) - ux ** 2 - uy ** 2
    
        # Check for unphysical particles with uz**2 < 0
        mask = uz_sqr >= 0
        N_new = np.count_nonzero(mask)
        if N_new < n_macroparticles:
            warnings.warn(
                  "Particles with uz**2<0 detected."
                  " %d Particles will be removed from the beam. \n"
                  "This will truncate the distribution of the beam"
                  " at gamma ~= 1. \n"
                  "However, the charge will be kept constant. \n"%(n_macroparticles
                                                                   - N_new))
            # Remove unphysical particles with uz**2 < 0
            x = x[mask]
            y = y[mask]
            z = z[mask]
            ux = ux[mask]
            uy = uy[mask]
            inv_gamma = inv_gamma[mask]
            uz_sqr = uz_sqr[mask]
        # Calculate longitudinal momentum of the bunch
        uz = np.sqrt(uz_sqr)
        # reverse uz if the beam is moving backwards
        if direction == 'backward':
            uz = -uz
        # Get weight of each particle
        w = n_physical_particles / N_new * np.ones_like(x)
        # Propagate distribution to an out-of-focus position tf.
        # (without taking space charge effects into account)
        if tf != 0.:
            x = x - ux * inv_gamma * c * tf
            y = y - uy * inv_gamma * c * tf
            z = z - uz * inv_gamma * c * tf
    
        # Save beam distribution to an .npz file
        if save_beam is not None:
            np.savez(save_beam, x=x, y=y, z=z, ux=ux, uy=uy, uz=uz,
                inv_gamma=inv_gamma, w=w)
            
        # end of shamelessly lifted code
    # broadcast the beam to other procs
    if sim.comm.size > 1:
        comm = sim.comm.mpi_comm
        
        if comm.rank != 0:
            x = np.empty(n_macroparticles)
            y = np.empty_like(x)
            z = np.empty_like(x)
            ux = np.empty_like(x)
            uy = np.empty_like(x)
            uz = np.empty_like(x)
            w = np.empty_like(x)
        
        x = comm.bcast(x, root=0)
        y = comm.bcast(y, root=0)
        z = comm.bcast(z, root=0)
        ux = comm.bcast(ux, root=0)
        uy = comm.bcast(uy, root=0)
        uz = comm.bcast(uz, root=0)
        w = comm.bcast(w, root=0)
        
    cathode = add_cathode_from_arrays(sim,q,m, z_cathode, x,y,z,ux,uy,uz,w,
                               initialize_self_field=initialize_self_field,
                               direction=direction, t_min=t_min, t_max=t_max,
                               rmax=rmax,
                               checkpoint_dir=checkpoint_dir,
                               checkpoint_period=checkpoint_period,
                               checkpoint_all_beam_info=checkpoint_all_beam_info)
    return cathode

def add_cathode_from_npz(sim, q, m, z_cathode, fname,
                            t_min=-np.inf, t_max=np.inf, rmax=None, 
                            initialize_self_field=True,
                            direction='forward',
                            checkpoint_dir='./checkpoints',
                            checkpoint_period=None,
                            checkpoint_all_beam_info=True):
    
    with np.load(fname) as beam:

        cathode = add_cathode_from_arrays(sim,q,m, z_cathode, 
                                beam['x'],beam['y'],beam['z'],
                                beam['ux'],beam['uy'],beam['uz'],beam['w'],
                                initialize_self_field=initialize_self_field,
                                direction=direction, t_min=t_min, t_max=t_max,
                                rmax=rmax,
                                checkpoint_dir=checkpoint_dir,
                                checkpoint_period=checkpoint_period,
                                checkpoint_all_beam_info=checkpoint_all_beam_info)
    return cathode

def add_cathode_from_arrays(sim, q, m, z_cathode, x,y,z,ux,uy,uz,w,
                            t_min=-np.inf, t_max=np.inf, rmax=None, 
                            initialize_self_field=True,
                            direction='forward',
                            checkpoint_dir='./checkpoints',
                            checkpoint_period=None,
                            checkpoint_all_beam_info=True):
    
    cathode = Cathode(sim,q,m, z_cathode, x,y,z,ux,uy,uz,w,
                               initialize_self_field=initialize_self_field,
                               direction=direction, t_min=t_min, t_max=t_max,
                               rmax=rmax,
                               checkpoint_dir=checkpoint_dir,
                               checkpoint_period=checkpoint_period,
                               checkpoint_all_beam_info=checkpoint_all_beam_info)
    sim.ptcl.append( cathode )
    sim.diags.append( cathode )
    return cathode
