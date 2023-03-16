# CUDA setup first
# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed, \
    cupy_installed, cupy_version, numba_cuda_installed
if cuda_installed:
    from fbpic.utils.cuda import send_data_to_gpu, receive_data_from_gpu
    if cupy_installed:
        import cupy

# standard modules
import sys, signal
# fbpic modules
from fbpic.utils.mpi import comm as mpi_comm
from fbpic.utils.mpi import MPI
from fbpic.main import Simulation as SimulationParent
from fbpic.utils.printing import ProgressBar
from fbpic.openpmd_diag import set_periodic_checkpoint

# define the SIGTERM handler class
class SigtermHandler:
    def __init__(self):
        self.sigterm_recieved = False 
        signal.signal( signal.SIGTERM, self.handle )
        #signal.signal( signal.SIGINT, self.handle )
    def handle(self, *args):
        print( 'SIGTERM recieved', flush=True )
        self.sigterm_recieved = True
# create an instance
sigterm_handler = SigtermHandler()

class Simulation(SimulationParent):
    
    # modified init
    def __init__(self, *args, t0=0, **kwargs):
        SimulationParent.__init__(self, *args, **kwargs)
        
        # set an arbitrary start time
        self.time = t0
        
    # modified step method
    def step(self, N=1, correct_currents=True,
             correct_divE=False, use_true_rho=False,
             move_positions=True, move_momenta=True, show_progress=True,
             write_termination_checkpoint=False):
        """
        Perform N PIC cycles.

        Parameters
        ----------
        N: int, optional
            The number of timesteps to take
            Default: N=1

        correct_currents: bool, optional
            Whether to correct the currents in spectral space

        correct_divE: bool, optional
            Whether to correct the divergence of E in spectral space

        use_true_rho: bool, optional
            Whether to use the true rho deposited on the grid for the
            field push or not. (requires initialize_ions = True)

        move_positions: bool, optional
            Whether to move or freeze the particles' positions

        move_momenta: bool, optional
            Whether to move or freeze the particles' momenta

        show_progress: bool, optional
            Whether to show a progression bar
            
        write_termination_checkpoint: bool, optional
            Whether to write a checkpoint in the even a SIGTERM is invoked
        """
        # Shortcuts
        ptcl = self.ptcl
        fld = self.fld
        dt = self.dt
        # Sanity check
        if self.comm.size > 1 and correct_divE:
            raise ValueError('correct_divE cannot be used in multi-proc mode.')
        if self.comm.size > 1 and use_true_rho and correct_currents:
            raise ValueError('`use_true_rho` cannot be used together '
                            'with `correct_currents` in multi-proc mode.')
            # This is because use_true_rho requires the guard cells of
            # rho to be exchanged while correct_currents requires the opposite.

        # Initialize the positions for continuous injection by moving window
        if self.comm.moving_win is not None:
            for species in self.ptcl:
                if species.continuous_injection:
                    species.injector.initialize_injection_positions(
                        self.comm, self.comm.moving_win.v, species.z, self.dt )

        # Initialize variables to measure the time taken by the simulation
        if show_progress and self.comm.rank==0:
            progress_bar = ProgressBar( N )

        # Send simulation data to GPU (if CUDA is used)
        if self.use_cuda:
            send_data_to_gpu(self)

        # Get the E and B fields in spectral space initially
        # (In the rest of the loop, E and B will only be transformed
        # from spectal space to real space, but never the other way around)
        self.comm.exchange_fields(fld.interp, 'E', 'replace')
        self.comm.exchange_fields(fld.interp, 'B', 'replace')
        self.comm.damp_EB_open_boundary( fld.interp )
        fld.interp2spect('E')
        fld.interp2spect('B')
        if self.use_pml:
            fld.interp2spect('E_pml')
            fld.interp2spect('B_pml')

        # Beginning of the N iterations
        # -----------------------------

        # Loop over timesteps
        for i_step in range(N):
            
            print(i_step, sigterm_handler.sigterm_recieved, flush=True)
            
            if not sigterm_handler.sigterm_recieved:
            
                # Show a progression bar and calculate ETA
                if show_progress and self.comm.rank==0:
                    progress_bar.time( i_step )
                    progress_bar.print_progress()
    
                # Particle exchanges to prepare for this iteration
                # ------------------------------------------------
    
                # Check whether this iteration involves particle exchange.
                # Note: Particle exchange is imposed at the first iteration
                # of this loop (i_step == 0) in order to ensure that all
                # particles are inside the box, and that 'rho_prev' is correct
                if self.iteration % self.comm.exchange_period == 0 or i_step == 0:
                    # Particle exchange includes MPI exchange of particles, removal
                    # of out-of-box particles and (if there is a moving window)
                    # continuous injection of new particles by the moving window.
                    # (In the case of single-proc periodic simulations, particles
                    # are shifted by one box length, so they remain inside the box)
                    for species in self.ptcl:
                        if hasattr(species, '_is_cathode'): 
                            species.inject_particles(self.time) 
                        self.comm.exchange_particles(species, fld, self.time)
                    for antenna in self.laser_antennas:
                        antenna.update_current_rank(self.comm)
    
                    # Reproject the charge on the interpolation grid
                    # (Since particles have been removed / added to the simulation;
                    # otherwise rho_prev is obtained from the previous iteration.)
                    self.deposit('rho_prev', exchange=(use_true_rho is True))
    
                    # For simulations on GPU, clear the memory pool used by cupy.
                    if self.use_cuda:
                        mempool = cupy.get_default_memory_pool()
                        mempool.free_all_blocks()
    
                # For the field diagnostics of the first step: deposit J
                # (Note however that this is not the *corrected* current)
                if i_step == 0:
                    self.deposit('J', exchange=True)
    
                # Main PIC iteration
                # ------------------
    
                # Keep field arrays sorted throughout gathering+push
                for species in ptcl:
                    species.keep_fields_sorted = True
    
                # Gather the fields from the grid at t = n dt
                for species in ptcl:
                    species.gather( fld.interp, self.comm )
                # Apply the external fields at t = n dt
                for ext_field in self.external_fields:
                    ext_field.apply_expression( self.ptcl, self.time )
    
                # Run the diagnostics
                # (after gathering ; allows output of gathered fields on particles)
                # (E, B, rho, x are defined at time n ; J, p at time n-1/2)
                for diag in self.diags:
                    # Check if the diagnostic should be written at this iteration
                    # (If needed: bring rho/J from spectral space, where they
                    # were smoothed/corrected, and copy the data from the GPU.)
                    diag.write( self.iteration )
    
                # Push the particles' positions and velocities to t = (n+1/2) dt
                if move_momenta:
                    for species in ptcl:
                        species.push_p( self.time + 0.5*self.dt )
                if move_positions:
                    for species in ptcl:
                        species.push_x( 0.5*dt )
                # Get positions/velocities for antenna particles at t = (n+1/2) dt
                for antenna in self.laser_antennas:
                    antenna.update_v( self.time + 0.5*dt )
                    antenna.push_x( 0.5*dt )
                # Shift the boundaries of the grid for the Galilean frame
                if self.use_galilean:
                    self.shift_galilean_boundaries( 0.5*dt )
    
                # Handle elementary processes at t = (n + 1/2)dt
                # i.e. when the particles' velocity and position are synchronized
                # (e.g. ionization, Compton scattering, ...)
                for species in ptcl:
                    species.handle_elementary_processes( self.time + 0.5*dt )
    
                # Fields are not used beyond this point ; no need to keep sorted
                for species in ptcl:
                    species.keep_fields_sorted = False
    
                # Get the current at t = (n+1/2) dt
                # (Guard cell exchange done either now or after current correction)
                self.deposit('J', exchange=(correct_currents is False))
                # Perform cross-deposition if needed
                if correct_currents and fld.current_correction=='cross-deposition':
                    self.cross_deposit( move_positions )
    
                # Push the particles' positions to t = (n+1) dt
                if move_positions:
                    for species in ptcl:
                        species.push_x( 0.5*dt )
                # Get positions for antenna particles at t = (n+1) dt
                for antenna in self.laser_antennas:
                    antenna.push_x( 0.5*dt )
                # Shift the boundaries of the grid for the Galilean frame
                if self.use_galilean:
                    self.shift_galilean_boundaries( 0.5*dt )
    
                # Get the charge density at t = (n+1) dt
                self.deposit('rho_next', exchange=(use_true_rho is True))
                # Correct the currents (requires rho at t = (n+1) dt )
                if correct_currents:
                    fld.correct_currents( check_exchanges=(self.comm.size > 1) )
                    if self.comm.size > 1:
                        # Exchange the guard cells of corrected J between domains
                        # (If correct_currents is False, the exchange of J
                        # is done in the function `deposit`)
                        fld.spect2partial_interp('J')
                        self.comm.exchange_fields(fld.interp, 'J', 'add')
                        fld.partial_interp2spect('J')
                    fld.exchanged_source['J'] = True
    
                # Push the fields E and B on the spectral grid to t = (n+1) dt
                fld.push( use_true_rho, check_exchanges=(self.comm.size > 1) )
                if correct_divE:
                    fld.correct_divE()
                # Move the grids if needed
                if self.comm.moving_win is not None:
                    # Shift the fields is spectral space and update positions of
                    # the interpolation grids
                    self.comm.move_grids(fld, ptcl, dt, self.time)
    
                # Handle boundaries for the E and B fields:
                # - MPI exchanges for guard cells
                # - Damp fields in damping cells
                # - Set fields to 0 at the position of the mirrors
                # - Update the fields in interpolation space
                #  (needed for the field gathering at the next iteration)
                self.exchange_and_damp_EB()
    
                # Increment the global time and iteration
                self.time += dt
                self.iteration += 1
    
                # Write the checkpoints if needed
                for checkpoint in self.checkpoints:
                    checkpoint.write( self.iteration )
            
            else: # if a sigterm _has_ been recieved
                print( 'PIC loop diverted', self.comm.rank, flush=True )
                
                if write_termination_checkpoint:
                    print( 'writing termination checkpoint...', self.comm.rank, flush=True )
                    # clear and replace any checkpoints
                    self.checkpoints = []
                    set_periodic_checkpoint(self, 1)
                    print( '  checkpoint reset', self.comm.rank, flush=True )
                    print( self.checkpoints, self.comm.rank, flush=True )
                    for checkpoint in self.checkpoints:
                        print( '  iterating...', self.comm.rank, flush=True )
                        checkpoint.write( self.iteration )

                    print( 'checkpoint written', self.comm.rank, flush=True )
                    
                #sys.exit(0)
                #print('this should not print', self.comm.rank, flush=True )
                return
            
        # End of the N iterations
        # -----------------------
        
        # Finalize PIC loop
        # Get the charge density and the current from spectral space.
        fld.spect2interp('J')
        if (not fld.exchanged_source['J']) and (self.comm.size > 1):
            self.comm.exchange_fields(self.fld.interp, 'J', 'add')
        fld.spect2interp('rho_prev')
        if (not fld.exchanged_source['rho_prev']) and (self.comm.size > 1):
            self.comm.exchange_fields(self.fld.interp, 'rho', 'add')

        # Receive simulation data from GPU (if CUDA is used)
        if self.use_cuda:
            receive_data_from_gpu(self)

        # Print the measured time taken by the PIC cycle
        if show_progress and (self.comm.rank==0):
            progress_bar.print_summary()
