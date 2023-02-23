# Copyright 2020, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Thomas Wilson
# License: 3-Clause-BSD-LBNL

"""
This file defines extensions to the FBPIC Particles class
"""
import warnings
import numpy as np

from fbpic.utils.cuda import cuda_installed
if cuda_installed:
    # Load the CUDA method
    import cupy
    from fbpic.utils.cuda import cuda_gpu_model

from fbpic.main import adapt_to_grid
from fbpic.particles import Particles
from fbpic.particles.injection.continuous_injection import unalign_angles, \
                                                generate_evenly_spaced, \
                                                _check_dens_func_arguments, \
                                                ContinuousInjector

def generate_evenly_spaced_dense( Npz, zmin, zmax, dr_sim, p_nr, Npr, rmin, rmax,
    Nptheta, n, dens_func, ux_m, uy_m, uz_m, ux_th, uy_th, uz_th, 
    weighting_method='asymptotic', unalign_angles_method='irrational' ):
    """
    Generate dense macroparticles with a configurable weighting distribution, 
    modulated according to the density function `dens_func`, and with the 
    momenta given by the `ux/y/z` arguments.

    Parameters
    ----------
    See the docstring of the `Particles` object for most options.
    
    dr_sim: float
        Simulation radial grid spacing
    
    p_nr
        particles-per-cell in r
        
    weighting_method: str, optional
        Weighting method, one of (`asymptotic`, `uniform', `quantised`)
        Default: `asymptotic`
        
    unalign_angles_method: str, optional
        Particle unalignment

    """

    # Generate the particles and eliminate the ones that have zero weight ;
    # infer the number of particles Ntot
    if Npz*Npr*Nptheta > 0:
        # Get the 1d arrays of evenly-spaced positions for the particles
        dz = (zmax-zmin)*1./Npz
        z_reg =  zmin + dz*( np.arange(Npz) + 0.5 )
        dr = (rmax-rmin)*1./Npr
        r_reg =  rmin + dr*( np.arange(Npr) + 0.5 )
        
        # To maintain roughly equal weights, each cell out from the axis must contain 
        # 2n-1 times more particles than the first cell. To do this, we 
        # duplicate the particles in r to steadily increase the number of 
        # particles per cell around the axis, thus maintaining the gridloading
        
        # Get the first and last radial cells in which particles appear to 
        # determine the number of repeats for each position in r_reg
        r_cell_min = r_reg[0]//dr_sim 
        
        repeats = int(r_cell_min + 1) + np.arange( len(r_reg)//p_nr, dtype=int ) 
        repeats = 2*repeats - 1 
        # Duplicate the values in these arrays
        repeats = np.repeat( repeats, p_nr)
        
        # Perform the duplication of r_reg and update Npr
        r_reg = np.repeat( r_reg, repeats )
        Npr = len(r_reg)
        # Also duplicate the repeats to later correct the particle weights 
        repeats = np.repeat(repeats, repeats )

        dtheta = 2*np.pi/Nptheta
        theta_reg = dtheta * np.arange(Nptheta)
        # Get the corresponding particles positions
        # (copy=True is important here, since it allows to
        # change the angles individually)
        zp, rp, thetap = np.meshgrid( z_reg, r_reg, theta_reg,
                                    copy=True, indexing='ij' )
        
        # Get the weights (i.e. charge of each macroparticle)
        # each method will produce the same total charge
        if weighting_method == 'uniform':
            wp = np.ones_like(rp) * n * np.pi*(rmax**2-rmin**2)*(zmax-zmin) / rp.size
        elif weighting_method == 'quantised':
            cells = repeats//2
            wp = n * (rp - cells[None,:,None]*dr*p_nr) * dtheta*dr*dz
        elif weighting_method == 'asymptotic':            
            wp = n * rp * dtheta*dr*dz
            wp /= repeats[None,:,None]
        else:
            raise ValueError(f"`weighting_method` expected one of 'uniform', 'quantised' or 'asymptotic', but got '{weighting_method}'")
            
        # Prevent the particles from being aligned along any direction
        unalign_angles( thetap, Npz, Npr, method=unalign_angles_method )
        # Flatten them (This performs a memory copy)
        r = rp.flatten()
        x = r * np.cos( thetap.flatten() )
        y = r * np.sin( thetap.flatten() )
        z = zp.flatten()
        w = wp.flatten()

        # Modulate it by the density profile
        if dens_func is not None :
            args = _check_dens_func_arguments( dens_func )
            if args == ['x', 'y', 'z']:
                w *= dens_func( x=x, y=y, z=z )
            elif args == ['z', 'r']:
                w *= dens_func( z=z, r=r )

        # Select the particles that have a non-zero weight
        selected = (w > 0)
        if np.any(w < 0):
            warnings.warn(
            'The specified particle density returned negative densities.\n'
            'No particles were generated in areas of negative density.\n'
            'Please check the validity of the `dens_func`.')

        # Infer the number of particles and select them
        Ntot = int(selected.sum())
        x = x[ selected ]
        y = y[ selected ]
        z = z[ selected ]
        w = w[ selected ]
        # Initialize the corresponding momenta
        uz = uz_m * np.ones(Ntot) + uz_th * np.random.normal(size=Ntot)
        ux = ux_m * np.ones(Ntot) + ux_th * np.random.normal(size=Ntot)
        uy = uy_m * np.ones(Ntot) + uy_th * np.random.normal(size=Ntot)
        inv_gamma = 1./np.sqrt( 1 + ux**2 + uy**2 + uz**2 )
        # Return the particle arrays
        return( Ntot, x, y, z, ux, uy, uz, inv_gamma, w )
    else:
        # No particles are initialized ; the arrays are still created
        Ntot = 0
        return( Ntot, np.empty(0), np.empty(0), np.empty(0), np.empty(0),
                      np.empty(0), np.empty(0), np.empty(0), np.empty(0) )

class DenseContinuousInjector(ContinuousInjector):
    """
    Class that stores a number of attributes that are needed for
    continuous injection by a moving window of the extended Particles class
    """
    def __init__(self, Npz, zmin, zmax, dz_particles, Npr, rmin, rmax,
                Nptheta, n, dens_func, ux_m, uy_m, uz_m, ux_th, uy_th, uz_th,
                dr_sim, np_r, dense_macroparticles=True,
                weighting_method='asymptotic', 
                unalign_angles_method='irrational'):
        
        ContinuousInjector.__init__(self, Npz, zmin, zmax, dz_particles, Npr, rmin, rmax,
                Nptheta, n, dens_func, ux_m, uy_m, uz_m, ux_th, uy_th, uz_th )

        self.dr_sim = dr_sim
        self.np_r = np_r
        self.dense_macroparticles = dense_macroparticles
        self.unalign_angles_method = unalign_angles_method
        self.weighting_method = weighting_method
        return
        
        
    def generate_particles( self, time ):
        """
        Generate new particles at the right end of the plasma
        (i.e. between z_end_plasma - nz_inject*dz and z_end_plasma)

        Parameters
        ----------
        time: float (in second)
            The current physical time of the simulation
        """
        # Create a temporary density function that takes into
        # account the fact that the plasma has moved
        if self.dens_func is not None:
            args = _check_dens_func_arguments( self.dens_func )
            if args == ['z', 'r']:
                def dens_func(z, r):
                    return self.dens_func( z - self.v_end_plasma*time, r )
            elif args == ['x', 'y', 'z']:
                def dens_func(x, y, z):
                    return self.dens_func( x, y, z - self.v_end_plasma*time )
        else:
            dens_func = None

        # Create new particle cells
        # Determine the positions between which new particles will be created
        Npz = self.nz_inject
        zmax = self.z_end_plasma
        zmin = self.z_end_plasma - self.nz_inject*self.dz_particles
        
        # Create the particles, either sparse or dense 
        if self.dense_macroparticles:
            Ntot, x, y, z, ux, uy, uz, inv_gamma, w = generate_evenly_spaced_dense(
                    Npz, zmin, zmax, self.dr_sim, self.np_r, self.Npr, self.rmin, self.rmax,
                    self.Nptheta, self.n, dens_func,
                    self.ux_m, self.uy_m, self.uz_m,
                    self.ux_th, self.uy_th, self.uz_th, 
                    unalign_angles_method=self.unalign_angles_method,
                    weighting_method=self.weighting_method)
        else:
            Ntot, x, y, z, ux, uy, uz, inv_gamma, w = generate_evenly_spaced(
                    Npz, zmin, zmax, self.Npr, self.rmin, self.rmax,
                    self.Nptheta, self.n, dens_func,
                    self.ux_m, self.uy_m, self.uz_m,
                    self.ux_th, self.uy_th, self.uz_th, 
                    method=self.unalign_angles_method)

        # Reset the number of particle cells to be created
        self.nz_inject = 0

        return( Ntot, x, y, z, ux, uy, uz, inv_gamma, w )
    
class DenseParticles(Particles):
    """
    Class that contains the particles data of the simulation.
    Extended to handle alternate macroparticle distributions.
    """
    def __init__(self, q, m, n, Npz, zmin, zmax,
                    Npr, rmin, rmax, Nptheta, dt,
                    dr_sim, p_nr,
                    ux_m=0., uy_m=0., uz_m=0.,
                    ux_th=0., uy_th=0., uz_th=0.,
                    dens_func=None, continuous_injection=True,
                    grid_shape=None, particle_shape='linear',
                    use_cuda=False, dz_particles=None,
                    dense_macroparticles=True, weighting_method='asymptotic',
                    unalign_angles_method='irrational' ):
        """
        See the docstring of the `Particles` object for most options
        See the docstring of `add_new_dense_species` for the new options
        """     
        # Define whether or not to use the GPU
        self.use_cuda = use_cuda
        if (self.use_cuda==True) and (cuda_installed==False) :
            warnings.warn(
                'Cuda not available for the particles.\n'
                'Performing the particle operations on the CPU.')
            self.use_cuda = False
        self.data_is_on_gpu = False # Data is initialized on the CPU

        # Generate evenly-spaced particles, either sparse or dense
        if dense_macroparticles:
            Ntot, x, y, z, ux, uy, uz, inv_gamma, w = generate_evenly_spaced_dense(
                Npz, zmin, zmax, dr_sim, p_nr, Npr, rmin, rmax, Nptheta, n, dens_func,
                ux_m, uy_m, uz_m, ux_th, uy_th, uz_th, 
                unalign_angles_method=unalign_angles_method,
                weighting_method=weighting_method)
        else:
            Ntot, x, y, z, ux, uy, uz, inv_gamma, w = generate_evenly_spaced( 
                Npz, zmin, zmax, Npr, rmin, rmax, Nptheta, n, dens_func, 
                ux_m, uy_m, uz_m, ux_th, uy_th, uz_th,
                method=unalign_angles_method,)
            
        # Register the properties of the particles
        # (Necessary for the pusher, and when adding more particles later, )
        self.Ntot = Ntot
        self.q = q
        self.m = m
        self.dt = dt

        # Register the particle arrarys
        self.x = x
        self.y = y
        self.z = z
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.inv_gamma = inv_gamma
        self.w = w

        # Initialize the fields array (at the positions of the particles)
        self.Ez = np.zeros( Ntot )
        self.Ex = np.zeros( Ntot )
        self.Ey = np.zeros( Ntot )
        self.Bz = np.zeros( Ntot )
        self.Bx = np.zeros( Ntot )
        self.By = np.zeros( Ntot )
        
        # register the unalignment method
        self.unalign_angles_method = unalign_angles_method
        # The particle injector stores information that is useful in order
        # continuously inject particles in the simulation, with moving window
        self.continuous_injection = continuous_injection
        if continuous_injection:
            self.injector = DenseContinuousInjector( Npz, zmin, zmax, dz_particles,
                                                Npr, rmin, rmax,
                                                Nptheta, n, dens_func,
                                                ux_m, uy_m, uz_m,
                                                ux_th, uy_th, uz_th,
                                                dr_sim, p_nr,
                                                unalign_angles_method=unalign_angles_method,
                                                weighting_method=weighting_method)
        else:
            self.injector = None

        # By default, there is no particle tracking (see method track)
        self.tracker = None
        # By default, the species experiences no elementary processes
        # (see method make_ionizable and activate_compton)
        self.ionizer = None
        self.compton_scatterer = None
        # Total number of quantities (necessary in MPI communications)
        self.n_integer_quantities = 0
        self.n_float_quantities = 8 # x, y, z, ux, uy, uz, inv_gamma, w

        # Register particle shape
        self.particle_shape = particle_shape

        # Register boolean that records whether field array should
        # be rearranged whenever sorting particles
        # (gets modified during the main PIC loop, on GPU)
        self.keep_fields_sorted = False

        # Allocate arrays and register variables when using CUDA
        if self.use_cuda:
            if grid_shape is None:
                raise ValueError("A `grid_shape` is needed when running "
                "on the GPU.\nPlease provide it when initializing particles.")
            # Register grid shape
            self.grid_shape = grid_shape
            # Allocate arrays for the particles sorting when using CUDA
            # Most required arrays always stay on GPU
            Nz, Nr = grid_shape
            self.cell_idx = cupy.empty( Ntot, dtype=np.int32)
            self.sorted_idx = cupy.empty( Ntot, dtype=np.intp)
            self.prefix_sum = cupy.empty( Nz*(Nr+1), dtype=np.int32 )
            # sorting buffers are initialized on CPU like other particle arrays
            # (because they are swapped with these arrays during sorting)
            self.sorting_buffer = np.empty( Ntot, dtype=np.float64)

            # Register integer thta records shift in the indices,
            # induced by the moving window
            self.prefix_sum_shift = 0
            # Register boolean that records if the particles are sorted or not
            self.sorted = False
            # Define optimal number of CUDA threads per block for deposition
            # and gathering kernels (determined empirically)
            if particle_shape == "cubic":
                self.deposit_tpb = 32
                self.gather_tpb = 256
            else:
                self.deposit_tpb = 16 if cuda_gpu_model == "V100" else 8
                self.gather_tpb = 128  
    
    def make_ionizable(self, element, target_species, level_start=0, level_max=None):
        # 
        warnings.warn(f'Attempting to turn on ionisation for {self}.\nUnless you are using a modified copy of fbpic.particles.elementary_process.ionization.ionizer this code is about to crash!\n')
        Particles.make_ionizable(self, element, target_species, 
                                 level_start=level_start, level_max=level_max)
        return
    
def add_new_dense_species( sim, q, m, n=None, dens_func=None,
                            p_nz=None, p_nr=None, p_nt=None,
                            p_zmin=-np.inf, p_zmax=np.inf,
                            p_rmin=0, p_rmax=np.inf,
                            uz_m=0., ux_m=0., uy_m=0.,
                            uz_th=0., ux_th=0., uy_th=0.,
                            continuous_injection=True,
                            boost_positions_in_dens_func=False,
                            dense_macroparticles=True, weighting_method='asymptotic',
                            unalign_angles_method='irrational'):
        """
        Add a new species to the simulation, by default employing dense
        macroparticle distribution
        
        Parameters
        ----------
        See `Simulation.add_new_species` for most options
        
        dense_macroparticles: boolean, optional
            Wether or not to use a dense macroparticle distribution. If this
            is `False`, the resulting species is identical to a normal
            Particles object, albeit by a different name.
            
        weighting_method: str, optional
            The particle weighting method. Accepts the following values:
                
            `uniform` 
                Generate identically weighted particles.
                This causes a slight increase in density and current on-axis.
              
            `quantised` - Generate particles that repeat the weights of
                particles in the first cell. This means across all particles 
                there will be only `p_nr` discrete weights (before modulation).
                This causes a slight decrease in density and current on-axis.
              
            `asymptotic` - Use an adapted version of the standard particle
                initialisation routine. Particles in the first cell have the 
                same weights as the standard method, with particle weights in 
                successive cells asymptoting towards a uniform value as the 
                distance from the axis increases.
              
            Regardless of the method chosen, the same number of particles will
            be created, and the total charge of those particles will be the 
            same. Only the distribution of weights is affected.
            Default: `asymptotic`
            
        unalign_angles_method: str, optional
            Set the method used to unalign particles around the azimuth.
            See particles.injection.continuous_injection.unalign_angles.
            Default: `irrational`
        """
        # Define a temporary density function
        # (will be modified below, in the case `boost_positions_in_dens_func`)
        new_dens_func = dens_func

        # Check if any macroparticle need to be injected
        if n is not None:
            # Check that all required arguments are passed
            for var in [p_nz, p_nr, p_nt]:
                if var is None:
                    raise ValueError(
                    'If the density `n` is passed to `add_new_species`,\n'
                    'then the arguments `p_nz`, `p_nr` and `p_nt` need '
                    'to be passed too.')

            # Automatically convert input quantities to the boosted frame
            if sim.boost is not None:
                gamma_m = np.sqrt(1. + uz_m**2 + ux_m**2 + uy_m**2)
                beta_m_lab = uz_m/gamma_m
                # Transform positions and density
                p_zmin, p_zmax = sim.boost.copropag_length(
                    [ p_zmin, p_zmax ], beta_object=beta_m_lab )
                n, = sim.boost.copropag_density([ n ], beta_object=beta_m_lab )
                # Transform longitudinal thermal velocity
                # The formulas below are approximate, and are obtained
                # by perturbation of the Lorentz transform for uz
                if uz_m == 0:
                    if uz_th > 0.1:
                        warnings.warn(
                        "The thermal distribution is approximate in "
                        "boosted-frame simulations, and may not be accurate "
                        "enough for uz_th > 0.1")
                    uz_th = sim.boost.gamma0 * uz_th
                else:
                    if uz_th > 0.1 * uz_m:
                        warnings.warn(
                        "The thermal distribution is approximate in "
                        "boosted-frame simulations, and may not be accurate "
                        "enough for uz_th > 0.1 * uz_m")
                    uz_th = sim.boost.gamma0 * \
                            (1. - sim.boost.beta0*beta_m_lab) * uz_th
                # Finally transform the longitudinal momentum
                uz_m = sim.boost.gamma0*( uz_m - sim.boost.beta0*gamma_m )

                # Create a temporary density function
                # that takes into account the Lorentz boost of the positions
                # (The motion of the plasma is further taken into account
                # in continuous_injection.py.)
                if boost_positions_in_dens_func and (dens_func is not None):

                    coef = sim.boost.gamma0*(1 - beta_m_lab*sim.boost.beta0)
                    args = _check_dens_func_arguments( dens_func )
                    if args == ['z', 'r']:
                        def new_dens_func( z, r ):
                            return dens_func( coef*z, r )
                    elif args == ['x', 'y', 'z']:
                        def new_dens_func( x, y, z ):
                            return dens_func( x, y, coef*z )

            # Modify input particle bounds, in order to only initialize the
            # particles in the local sub-domain
            zmin_local_domain, zmax_local_domain = sim.comm.get_zmin_zmax(
                                        local=True, rank=sim.comm.rank,
                                        with_damp=False, with_guard=False )
            p_zmin = max( zmin_local_domain, p_zmin )
            p_zmax = min( zmax_local_domain, p_zmax )
            # Avoid that particles get initialized in the radial PML cells
            rmax = sim.comm.get_rmax( with_damp=False )
            p_rmax = min( rmax, p_rmax )

            # Modify again the input particle bounds, so that
            # they fall exactly on the grid, and infer the number of particles
            p_zmin, p_zmax, Npz = adapt_to_grid( sim.fld.interp[0].z,
                                p_zmin, p_zmax, p_nz )
            p_rmin, p_rmax, Npr = adapt_to_grid( sim.fld.interp[0].r,
                                p_rmin, p_rmax, p_nr )
            dz_particles = sim.comm.dz/p_nz

        else:
            # Check consistency of arguments
            if (dens_func is not None) or (p_nz is not None) or \
                (p_nr is not None) or (p_nt is not None):
                warnings.warn(
                    'It seems that you provided the arguments `dens_func`, '
                    '`p_nz`, `p_nr` or `p_nz`\nHowever no particle density '
                    '(`n` or `n_e`) was given.\nTherefore, no particles will'
                    'be created.')
            # Convert arguments to acceptable arguments for `Particles`
            # but which will result in no macroparticles being injected
            n = 0
            p_zmin = p_zmax = p_rmin = p_rmax = 0
            Npz = Npr = p_nt = 0
            continuous_injection = False
            dz_particles = 0.

        # Create the new species
        new_species = DenseParticles( q=q, m=m, n=n, dens_func=new_dens_func,
                        Npz=Npz, zmin=p_zmin, zmax=p_zmax,
                        Npr=Npr, rmin=p_rmin, rmax=p_rmax,
                        Nptheta=p_nt, dt=sim.dt,
                        dr_sim=sim.comm.dr, p_nr=p_nr,
                        particle_shape=sim.particle_shape,
                        use_cuda=sim.use_cuda, grid_shape=sim.grid_shape,
                        ux_m=ux_m, uy_m=uy_m, uz_m=uz_m,
                        ux_th=ux_th, uy_th=uy_th, uz_th=uz_th,
                        continuous_injection=continuous_injection,
                        dz_particles=dz_particles,
                        dense_macroparticles=dense_macroparticles, weighting_method=weighting_method,
                        unalign_angles_method=unalign_angles_method)

        # Add it to the list of species and return it to the user
        sim.ptcl.append( new_species )
        return new_species

#%% Testing
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from fbpic.main import Simulation
    from scipy.constants import e, m_e
    
    dens_func = lambda z,r: np.ones_like(z)
   
    zmin = 0
    zmax = 1
    Nz = 1
    rmax = 10
    rmin = 0.
    Nr = 10
    Nm = 1
    dt = .000000001
    
    p_zmin = zmin
    p_zmax = zmax
    p_rmin = 0
    p_rmax = 10
    
    p_nz = 1
    p_nr = 4
    p_nt = 4
    
    n = 1.

    unalign_angles_method='irrational'
    weighting_method = 'asymptotic'
    


    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin, 
                     boundaries={'z':'open', 'r':'reflective'})

    elec0 = add_new_dense_species(sim, -e, m_e, n=n, dens_func=dens_func, 
                                         p_nz=p_nz, p_nr=p_nr, p_nt=p_nt, 
                                         p_zmin=p_zmin, p_zmax=p_zmax, 
                                         p_rmin=p_rmin, p_rmax=p_rmax, 
                                         unalign_angles_method=unalign_angles_method,
                                         dense_macroparticles=False)
    x0 = elec0.x
    y0 = elec0.y
    w0 = elec0.w
    r0 = np.sqrt( x0**2 + y0**2 )
    th0 = np.arctan2(y0,x0)
    
    elec = add_new_dense_species(sim, -e, m_e, n=n, dens_func=dens_func, 
                                         p_nz=p_nz, p_nr=p_nr, p_nt=p_nt, 
                                         p_zmin=p_zmin, p_zmax=p_zmax, 
                                         p_rmin=p_rmin, p_rmax=p_rmax, 
                                         unalign_angles_method=unalign_angles_method,
                                         weighting_method=weighting_method)
    x = elec.x
    y = elec.y
    w = elec.w
    r = np.sqrt( x**2 + y**2 )
    th = np.arctan2(y,x)
    

    fig, axes = plt.subplots( figsize=(12,5), constrained_layout=False, ncols=2,
                           subplot_kw={'projection':'polar'})
    
    rtheta = np.linspace( 0, 2*np.pi, 1000)
    
    ax = axes[0]
    
    for ring in np.linspace( rmin, rmax, Nr+1 )[1:-1]:
        ax.plot(rtheta, np.full_like(rtheta, ring), 'k:', lw=1 )

    ax.set_thetagrids(angles=[], labels=[])
    ax.set_rgrids(radii=[], labels=[])
    ax.set_yticklabels([])
    
    #ax.plot( w )
    img = ax.scatter( th, r, c=w, cmap='cool', s=w*10, vmin=0.05, vmax=0.35 )
    #ax.scatter( x0, y0, s=5, c='r' )
    cbar=fig.colorbar( img, ax=ax, label='weight [arb.]' )
    thetasymbol = r'\theta'
    ax.set_xlabel(f'$n_z={p_nz}$ $n_r={p_nr}$ $n_{thetasymbol}={p_nt}$ $N = {elec.Ntot}$\nmethod: {weighting_method}/{unalign_angles_method}')
    ax.set_ylim(0,rmax)
    #plt.plot( sorted(w) )

    ax = axes[1]

    for ring in np.linspace( rmin, rmax, Nr+1 )[1:-1]:
        ax.plot(rtheta, np.full_like(rtheta, ring), 'k:', lw=1 )

    ax.set_thetagrids(angles=[], labels=[])
    ax.set_rgrids(radii=[], labels=[])
    ax.set_yticklabels([])
    
    img = ax.scatter( th0, r0, c=w0, cmap='winter', s=w0*10 )
    #ax.scatter( x0, y0, s=5, c='r' )
    cbar=fig.colorbar( img, ax=ax, label='weight [arb.]' )
    thetasymbol = r'\theta'
    ax.set_xlabel(f'$n_z={p_nz}$ $n_r={p_nr}$ $n_{thetasymbol}={p_nt}$ $N = {elec0.Ntot}$\nmethod: standard/{unalign_angles_method}')
    ax.set_ylim(0,rmax)
    #plt.plot( sorted(w) )

    fig.tight_layout()
    
    print( w.sum(), w0.sum() )
    print( w[:p_nr])
    print( w0[:p_nr] )
    
    #fig.savefig(f"D:/work/fbpic/lwfa_example/weights_{weighting_method.replace(' ','_')}_{unalign_angles_method}.png", dpi=600)
    
    
    
    