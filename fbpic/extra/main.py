# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters, Soeren Jalas
# License: 3-Clause-BSD-LBNL
"""
Fourier-Bessel Particle-In-Cell (FB-PIC) main file

This file steers and controls the simulation.
"""
# When cuda is available, select one GPU per mpi process
# (This needs to be done before the other imports,
# as it sets the cuda context)
from fbpic.utils.mpi import MPI
# Check if threading is available
from fbpic.utils.threading import threading_enabled, numba_version
# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed, \
    cupy_installed, cupy_version, numba_cuda_installed
if cuda_installed:
    from fbpic.utils.cuda import send_data_to_gpu, \
                receive_data_from_gpu, mpi_select_gpus
    mpi_select_gpus( MPI )
    if cupy_installed:
        import cupy

# Import the rest of the requirements
import sys, signal
import warnings
import numba
import numpy as np
from scipy.constants import m_e, m_p, e, c
from fbpic.main import Simulation as SimulationParent
from fbpic.openpmd_diag import set_periodic_checkpoint
from fbpic.utils.printing import ProgressBar, print_simulation_setup
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.fields import Fields
from fbpic.boundaries import BoundaryCommunicator

# define the signal handler class
class SignalHandler:
    def __init__(self):
        """
        Class to handle a signal
        """
        self.num = None
        self.signal_recieved = False 
    
    def set_signal(self, num):
        """
        Create a flag for a specific signal code
        
        SIGTERM : 15
        SIGINT  : 2
        SIGUSR1 : 10    (unix only)
        """
        self.num = num
        signal.signal( num, self._handle )
        
    def _handle(self, *args):
        self.signal_recieved = True

class Simulation(SimulationParent):
    
    # modified init
    def __init__(self, Nz, zmax, Nr, rmax, Nm, dt,
                 p_zmin=-np.inf, p_zmax=np.inf, p_rmin=0, p_rmax=np.inf,
                 p_nz=None, p_nr=None, p_nt=None, n_e=None, zmin=0.,
                 n_order=-1, dens_func=None, filter_currents=True,
                 v_comoving=None, use_galilean=True,
                 initialize_ions=False, use_cuda=False, n_guard=None,
                 n_damp={'z':64, 'r':32},
                 exchange_period=None,
                 current_correction='curl-free',
                 boundaries={'z':'periodic', 'r':'reflective'},
                 gamma_boost=None, use_all_mpi_ranks=True,
                 particle_shape='linear', verbose_level=1,
                 smoother=None, use_ruyten_shapes=True,
                 use_modified_volume=True, 
                 t0=0., i0=0, shutdown_signal=15, 
                 custom_domain_decomposition=None):
        """
        New Parameters
        --------------
        t0: float, optional
            Set a start time other than zero.
        
        i0: int, optional
            Set a start iteration other than zero.
            
        shutdown_signal: int or signal, optional
            The signal used to trigger early shutdown. 
            Defaults to SIGUSR1 (15)
        """
        # Check whether to use CUDA
        self.use_cuda = use_cuda
        if self.use_cuda and not cuda_installed:
            warning_message = 'GPU not available for the simulation.\n'
            if not numba_cuda_installed:
                warning_message += \
                '(This is because the `numba` package was not able to find a GPU.)\n'
            elif not cupy_installed:
                warning_message += \
                '(This is because the `cupy` package is not installed.)\n'
            warning_message += 'Performing the simulation on CPU.'
            warnings.warn( warning_message )
            self.use_cuda = False
        # Check that cupy, numba and Python have the right version
        if self.use_cuda:
            if cupy_version < (7,0):
                raise RuntimeError(
                    'In order to run on GPUs, FBPIC version 0.20 and later \n'
                    'requires `cupy` version 7.0 (or later).\n(The `cupy` '
                    'version on your current system is %d.%d.)\nPlease '
                    'install the latest version of `cupy`.' %cupy_version)
            elif numba_version < (0,46):
                raise RuntimeError(
                    'In order to run on GPUs, FBPIC version 0.16 and later \n'
                    'requires `numba` version 0.46 (or later).\n(The `numba` '
                    'version on your current system is %d.%d.)\nPlease install'
                    ' the latest version of `numba`.' %numba_version)
            elif sys.version_info.major < 3:
                raise RuntimeError(
                    'In order to run on GPUs, FBPIC version 0.16 and later \n'
                    'requires Python 3.\n(The Python version on your current '
                    'system is Python 2.)\nPlease install Python 3.')
        # CPU multi-threading
        self.use_threading = threading_enabled
        if self.use_threading:
            self.cpu_threads = numba.config.NUMBA_NUM_THREADS
        else:
            self.cpu_threads = 1

        # Register the comoving parameters
        self.v_comoving = v_comoving
        self.use_galilean = use_galilean
        if v_comoving is None:
            self.use_galilean = False

        # When running the simulation in a boosted frame, convert the arguments
        if gamma_boost is not None:
            self.boost = BoostConverter( gamma_boost )
            zmin, zmax, dt = self.boost.copropag_length([ zmin, zmax, dt ])
        else:
            self.boost = None
        # Register time step
        self.dt = dt

        # Initialize the boundary communicator
        cdt_over_dr = c*dt / (rmax/Nr)
        self.comm = BoundaryCommunicator( Nz, zmin, zmax, Nr, rmax, Nm, dt,
            self.v_comoving, self.use_galilean, boundaries, n_order,
            n_guard, n_damp, cdt_over_dr, None, exchange_period,
            use_all_mpi_ranks, custom_domain_decomposition=custom_domain_decomposition )
        self.use_pml = self.comm.use_pml
        # Modify domain region
        zmin, zmax, Nz = self.comm.divide_into_domain()
        Nr = self.comm.get_Nr( with_damp=True )
        rmax = self.comm.get_rmax( with_damp=True )
        # Initialize the field structure
        self.fld = Fields( Nz, zmax, Nr, rmax, Nm, dt,
                    n_order=n_order, zmin=zmin,
                    v_comoving=v_comoving,
                    use_pml=self.use_pml,
                    use_galilean=use_galilean,
                    current_correction=current_correction,
                    use_cuda=self.use_cuda,
                    smoother=smoother,
                    # Only create threading buffers when running on CPU
                    create_threading_buffers=(self.use_cuda is False),
                    use_ruyten_shapes=use_ruyten_shapes,
                    use_modified_volume=use_modified_volume )

        # Initialize the electrons and the ions
        self.grid_shape = self.fld.interp[0].Ez.shape
        self.particle_shape = particle_shape
        self.ptcl = []
        if n_e is not None:
            # - Initialize the electrons
            self.add_new_species( q=-e, m=m_e, n=n_e, dens_func=dens_func,
                                  p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
                                  p_zmin=p_zmin, p_zmax=p_zmax,
                                  p_rmin=p_rmin, p_rmax=p_rmax )
            # - Initialize the ions
            if initialize_ions:
                self.add_new_species( q=e, m=m_p, n=n_e, dens_func=dens_func,
                                  p_nz=p_nz, p_nr=p_nr, p_nt=p_nt,
                                  p_zmin=p_zmin, p_zmax=p_zmax,
                                  p_rmin=p_rmin, p_rmax=p_rmax )
                

        # allow for an arbitrary start time and iteration
        self.time = t0
        self.iteration = i0
        # Register the filtering flag
        self.filter_currents = filter_currents

        # Initialize an empty list of external fields
        self.external_fields = []
        # Initialize an empty list of diagnostics and checkpoints
        # (Checkpoints are used for restarting the simulation)
        self.diags = []
        self.checkpoints = []
        # Initialize an empty list of laser antennas
        self.laser_antennas = []
        # Initialize an empty list of mirrors
        self.mirrors = []

        # Print simulation setup
        print_simulation_setup( self, verbose_level=verbose_level )

        # create the signal handler and set the trigger signal if specified
        self.handler = SignalHandler()
        if shutdown_signal is not None:   
            self.handler.set_signal(shutdown_signal)
        
    def write_exit_checkpoint(self):
        """ Immediately write a checkpoint """
        # create a copy of any pre-existing checkpoints first
        user_checkpoints = self.checkpoints[:]
        # clear the checkpoint list
        self.checkpoints = []
        # make the one-time checkpoint
        set_periodic_checkpoint(self, 1)
        for checkpoint in self.checkpoints:
            checkpoint.write( self.iteration )
        # swap the user checkpoints back in
        self.checkpoints = user_checkpoints[:]
        
    # modified step method
    def step(self, N=1, correct_currents=True,
             correct_divE=False, use_true_rho=False,
             move_positions=True, move_momenta=True, show_progress=True,
             write_exit_checkpoint=False):
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
            
        write_exit_checkpoint: bool, optional
            Whether to write a checkpoint in the event a SIGTERM is invoked
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
            
            # break the loop upon recipt of the signal
            if self.handler.signal_recieved: 
                break

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
            
        # End of the N iterations
        # -----------------------
        
        # write an exit checkpoint if needed
        if write_exit_checkpoint:
            self.write_exit_checkpoint()

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
