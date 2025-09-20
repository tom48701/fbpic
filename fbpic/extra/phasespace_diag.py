# Copyright 2020, FBPIC contributors
# Authors: Thomas Wilson, Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL

"""
This file defines the class PhaseSpaceDiagnostic and related CustomQuantity class
"""
import os, warnings
import h5py
import numpy as np
from scipy import constants
from fbpic.openpmd_diag.generic_diag import OpenPMDDiagnostic
from fbpic.openpmd_diag.data_dict import macro_weighted_dict, \
    weighting_power_dict, unit_dimension_dict

class CustomQuantity:
    """
    Class that defines a custom particle quantity to be calculated
    """
    def __init__(self, name, function, dimensions=np.zeros(7) ):
        """
        Initialise a custom quantity to calculate.

        Parameters
        ----------
        name : str
            Name of the quantity, used for reference.

        function : func
            Function to compute the custom quantity.
            The function takes argumentes named as per the internal FBPIC
            basic particle attributes.
            For example, the formula for the angular momentum
            component along z is (x*uy - y*ux), so a function of the form

            def lz_func(x, uy, y, ux):
            	return( x*uy - y*ux )

            would be suitable.

        dimensions : np.array of length 7, optional
            Dimensions of the quantity. If omitted, the result is
            assumed to be adimensional.
        """

        self.name = name
        self.function = function
        self.arguments = function.__code__.co_varnames[:function.__code__.co_argcount]
        self.dimensions = dimensions

    def __call__(self, *args):
        """
        Wrap the function call to ensure NoneType arguments are ignored.
        This is required to avoid crashes when running under MPI, when only
        the root process will recieve real data to work on
        """
        if any([arg is None for arg in args]):
            return None
        else:
            return self.function(*args)


# Useful custom quantities
# this list should eventually contain all predefined quantities for import
predefined_quantities = []
# From data_dict.py:
# Correspondance between quantity and corresponding dimensions
# As specified in the openPMD standard, the arrays represent the
# 7 basis dimensions L, M, T, I, theta, N, J

def rF(x,y):
    return np.sqrt(x**2+y**2)

def tF(x,y):
    return np.arctan2(y,x)

def eneF(gamma):
    return gamma - 1.

# radial position
radius_CQ = CustomQuantity( 'r', rF, unit_dimension_dict['position'] )
# azimuthal position
azimuth_CQ = CustomQuantity( 't', tF)
# normalised KE
energy_CQ = CustomQuantity( 'ene', eneF )

predefined_quantities += [radius_CQ, azimuth_CQ, energy_CQ]

# radial/azimuthal momentum
def urF(x,y,ux,uy):
    return (x*ux + y*uy) / np.sqrt(x**2 + y**2)

def utF(x,y,ux,uy):
    return (x*uy - y*ux) / np.sqrt(x**2 + y**2)

ur_CQ = CustomQuantity( 'ur', urF, unit_dimension_dict['momentum'] )
ut_CQ = CustomQuantity( 'ut', utF, unit_dimension_dict['momentum'] )

predefined_quantities += [ur_CQ, ut_CQ]

# angular momentum components
def lxF(y,uz,z,uy):
    return y*uz - z*uy

def lyF(x,uz,z,ux):
    return z*ux - x*uz

def lzF(x,uy,y,ux):
    return x*uy - y*ux

lx_CQ = CustomQuantity( 'lx', lxF, np.array([2,1,-1,0,0,0,0]) )
ly_CQ = CustomQuantity( 'ly', lyF, np.array([2,1,-1,0,0,0,0]) )
lz_CQ = CustomQuantity( 'lz', lzF, np.array([2,1,-1,0,0,0,0]) )

predefined_quantities += [lx_CQ, ly_CQ, lz_CQ]

# deflection angles
def zx_angleF(ux,uz):
    return np.arctan2( ux, uz )

def zy_angleF(uy,uz):
    return np.arctan2( uy, uz )

def zr_angleF(x,y,ux,uy,uz):
    ur = (x*ux + y*uy) / np.sqrt(x**2 + y**2)
    return np.arctan2( ur, uz )

zx_angle_CQ = CustomQuantity( 'zx_angle', zx_angleF )
zy_angle_CQ = CustomQuantity( 'zy_angle', zy_angleF )
zr_angle_CQ = CustomQuantity( 'zr_angle', zr_angleF )

predefined_quantities += [zx_angle_CQ, zy_angle_CQ, zr_angle_CQ]


class PhaseSpaceDiagnostic(OpenPMDDiagnostic) :
    """
    Class that defines a phase space diagnostic to be performed.
    """

    def __init__(self, period=None, species={}, comm=None,
        name=None, phase_space=[], bins=[], edges=None, custom_quantities=[],
        move_with_window=True, deposit='w', unweighted=False, select=None,
        write_dir=None, iteration_min=0, iteration_max=np.inf, dt_period=None,
        sim=None ) :
        """
        Initialize a phase space diagnostic.

        Parameters
        ----------
        period : int, optional
            The period of the diagnostic, in number of timesteps.
            (i.e. the diagnostic is written whenever the number
            of iterations is divisible by `period`). Specify either this or
            `dt_period`.

        dt_period : float (in seconds), optional
            The period of the diagnostic, in physical time of the simulation.
            Specify either this or `period`

        species : a dictionary of :any:`Particles` objects
            The object that is written (e.g. elec)
            is assigned to the particle name of this species.
            (e.g. {"electrons": elec })

        comm : an fbpic BoundaryCommunicator object or None
            If this is not None, the data is gathered by the communicator.
            Otherwise, each rank writes its own data,
            (Make sure to use different write_dir in this case).

        name: str, optional
            Specify a name to be used for the h5 dataset. If left as None, an
            automatically generated name is used based on the phase space
            and quantity deposited.

        phase_space : a list of strings
            Specify the phase space over which to bin particles.
            Allowed quantities are any combination of the basic particle
            quantities:
                position: 'x', 'y', 'z'
                momentum: 'uz', 'uy', 'uz'
                fields: 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz',
                Lorentz factor: 'gamma'
                weighting: 'w'
            as well as any custom quantities defined via `custom_quantities`.

        custom_quantities : a list of CustomQuantity instances, optional
            definitions for any additional custom quantities to be used in the
            diagnostic.
            These can be calculated using any of the basic particle
            quantities (listed above).

        bins: a list of ints
            A list of ints specifying the number of bins in each dimension.

        edges: list, optional
            Set the outer bin edges for each dimension.
            Leave as None to auto-scale, or set pairs of floats to specify
            limits, of the form.
            [[-1,1], None]  (set a range of (-1, 1) in the first dimension,
                             and auto-scale the second)
            When using a moving window, any limits for binning along z
            will automatically be shifted along with the box, this behaviour
            can be turned off with the move_with_window argument.

        move_with_window: bool, optional
            Move any given z limits along with the moving window

        deposit: string, optional
            Specify a particle quantity to deposit in the bins.
            The particle weight is always deposited.

        unweighted: bool, optional
            Set all particle weights to unity. Default: False

        select : dict, optional
            Either None or a dictionary of rules to select the particles, of
            the form
            'x' : [-4., 10.]   (Particles having x between -4 and 10 microns)
            'ux' : [-0.1, 0.1] (Particles having ux between -0.1 and 0.1 mc)
            'uz' : [5., None]  (Particles with uz above 5 mc)

        write_dir : a list of strings, optional
            The POSIX path to the directory where the results are
            to be written. If none is provided, this will be the path
            of the current working directory.

        iteration_min, iteration_max: ints
            The iterations between which data should be written
            (`iteration_min` is inclusive, `iteration_max` is exclusive)

        sim : fbpic Simulation object, optional
            Use this object to extract an unambiguous simulation time if
            provided
        """
        # Check input
        if len(species) == 0:
            raise ValueError("You need to pass an non-empty `species_dict`.")
        if len(phase_space) == 0:
            raise ValueError("You need to pass a non-empty `phase_space`")
        if move_with_window and comm is None:
            raise ValueError("A communicator is required for co-moving diagnostics")
        # Check the dimensions of the arguments
        if len(bins) != len(phase_space):
            raise ValueError("Dimensionality of the phase space and the bins must be equal.")
        if edges is not None:
            if len(edges) != len(phase_space):
                raise ValueError("Any edge limits specified must match the phase space dimensionality")

        # Build an ordered list of species. (This is needed since the order
        # of the keys is not well defined, so each MPI rank could go through
        # the species in a different order, if species_dict.keys() is used.)
        self.species_names_list = sorted( species.keys() )
        # Extract the timestep from the first species
        first_species = species[self.species_names_list[0]]
        self.dt = first_species.dt

        # General setup (uses the above timestep)
        OpenPMDDiagnostic.__init__(self, period, comm, write_dir,
                        iteration_min, iteration_max,
                        dt_period=dt_period, dt_sim=self.dt )

        # Register the arguments
        self.species_dict = species
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        self.name = name
        self.phase_space = phase_space
        self.select = select
        self.bins = bins
        self.edges = edges
        self.deposit = deposit
        self.unweighted = unweighted
        self.move_with_window = move_with_window
        self.sim = sim
        self.custom_quantities = predefined_quantities
        # prepend any user-defined custom quantities to the list
        if hasattr(custom_quantities, "__len__"):
            for quant in custom_quantities:
                self.custom_quantities.insert(0, quant)
        else:
            self.custom_quantities.append(custom_quantities)

        # Register the dimensionality of the diagnostic
        self.Ndims = len(phase_space)

        # storage of the minimum and maximum of each particle quantity
        self.quant_min = np.empty(self.Ndims)
        self.quant_max = np.empty(self.Ndims)

        # For each species, get the particle arrays to be written
        self.array_quantities_dict = {}

        for species_name in self.species_names_list:
            species = self.species_dict[species_name]
            # Get the list of the required particle quantities
            self.array_quantities_dict[species_name] = phase_space

        # register the initial zmin position for shifting the bins
        if move_with_window:
            self.zmin_init = self.comm._zmin_global_domain


    def setup_openpmd_phasespace_group( self, grp, species ):
        """
        Set the attributes that are specific to the phase spaces group
        Parameter
        ---------
        grp : an h5py.Group object
            Contains all the species
        species : a fbpic Particle object
        constant_quantities: list of strings
            The scalar quantities to be written for this particle
        """
        # Generic attributes
        grp.attrs["particleShape"] = 1.
        grp.attrs["currentDeposition"] = np.bytes_("directMorseNielson")
        grp.attrs["particleSmoothing"] = np.bytes_("none")
        grp.attrs["particlePush"] = np.bytes_("Vay")
        grp.attrs["particleInterpolation"] = np.bytes_("uniform")

    def setup_openpmd_species_record( self, grp, quantity ) :
        """
        Set the attributes that are specific to a species record
        Parameter
        ---------
        grp : an h5py.Group object or h5py.Dataset
            The group that correspond to `quantity`
            (in particular, its path must end with "/<quantity>")
        quantity : string
            The name of the record being setup
            e.g. "position", "momentum"
        """
        # Generic setup

        self.setup_openpmd_record( grp, quantity )

        # Weighting information
        grp.attrs["macroWeighted"] = macro_weighted_dict[quantity]
        grp.attrs["weightingPower"] = weighting_power_dict[quantity]

    def setup_openpmd_species_component( self, grp, quantity ) :
        """
        Set the attributes that are specific to a species component
        Parameter
        ---------
        grp : an h5py.Group object or h5py.Dataset
        quantity : string
            The name of the component
        """
        self.setup_openpmd_component( grp )

    def write_hdf5( self, iteration ) :
        """
        Write an HDF5 file that complies with the OpenPMD standard
        Parameter
        ---------
        iteration : int
             The current iteration number of the simulation.
        """
        # Receive data from the GPU if needed
        for species_name in self.species_names_list:
            species = self.species_dict[species_name]
            if species.use_cuda :
                species.receive_particles_from_gpu()

        # Create the file and setup the openPMD structure (only first proc)
        if self.rank == 0:
            filename = "data%08d.h5" %iteration
            fullpath = os.path.join( self.write_dir, "hdf5", filename )
            f = h5py.File( fullpath, mode="a" )

            # Setup its attributes
            if self.sim is None:
                time = iteration*self.dt
            else:
                time = self.sim.time
            self.setup_openpmd_file( f, iteration, time, self.dt)

        # Loop over the different species that should be written
        for species_name in self.species_names_list:

            # Check if the species exists
            species = self.species_dict[species_name]
            if species is None :
                # If not, immediately go to the next species_name
                continue

            # Setup the phasespace group (only first proc)
            if self.rank==0:
                phasespace_path = "/data/%d/phase_spaces/%s" %(
                    iteration, species_name)
                # Create and setup the h5py.Group phasespace_grp
                phasespace_grp = f.require_group( phasespace_path )
                self.setup_openpmd_phasespace_group( phasespace_grp, species )
            else:
                phasespace_grp = None

            # Select the particles that will be included
            select_array = self.apply_selection( species )

            # get the total size of the eventual histogram
            Ntot = np.prod(self.bins)

            # Write the dataset
            self.write_phasespaces(phasespace_grp, species, Ntot, select_array,
                                   self.array_quantities_dict[species_name])

        # Close the file
        if self.rank == 0:
            f.close()

        # Send data to the GPU if needed
        for species_name in self.species_names_list:
            species = self.species_dict[species_name]
            if species.use_cuda :
                species.send_particles_to_gpu()

    def write_phasespaces( self, phasespace_grp, species, Ntot,
                          select_array, phasespace_data ) :
        """
        Write all the phase space sets for one given species
        phasespace_grp : an h5py.Group
            The group where to write the species considered
        species : an fbpic.Particles object
        	The species object to get the particle data from
        n_rank : list of ints
            A list containing the number of particles to send on each proc
        Ntot : int
        	Contains the global number of particles
        select_array : 1darray of bool
            An array of the same shape as that particle array
            containing True for the particles that satify all
            the rules of self.select
        phasespace_data: list of string
            The phasespace quantities that should be written
        """
        # Set the dataset name if one was provided
        if self.name is not None:
            quantity_path = self.name
        else:
            # Otherwise automatically generate a name
            quantity_path = '%s_%s'%('_'.join(self.phase_space), self.deposit)

        self.write_dataset( phasespace_grp, species, quantity_path,
                            Ntot, select_array )


    def apply_selection( self, species ) :
        """
        Apply the rules of self.select to determine which
        particles should be written, Apply random subsampling using
        the property subsampling_fraction.
        Parameters
        ----------
        species : a Species object
        Returns
        -------
        A 1d array of the same shape as that particle array
        containing True for the particles that satify all
        the rules of self.select
        """
        # Initialize an array filled with True
        select_array = np.ones( species.Ntot, dtype='bool' )


        # Apply the rules successively
        if self.select is not None :
            # Go through the quantities on which a rule applies
            for quantity in self.select.keys() :
                if quantity == "gamma":
                    quantity_array = 1.0/getattr( species, "inv_gamma" )
                elif quantity == 'r':
                    quantity_array = np.sqrt(species.x**2 + species.y**2)
                else:
                    quantity_array = getattr( species, quantity )
                # Lower bound
                if self.select[quantity][0] is not None :
                    if quantity == 'z' and self.move_with_window:
                        # shift limits
                        zrel = self.select[quantity][0] - self.zmin_init
                        qmin = self.comm._zmin_global_domain + zrel
                    else:
                        qmin = self.select[quantity][0]

                    select_array = np.logical_and(
                        quantity_array > qmin,
                        select_array )
                # Upper bound
                if self.select[quantity][1] is not None :
                    if quantity == 'z' and self.move_with_window:
                        # shift limits
                        zrel = self.select[quantity][1] - self.zmin_init
                        qmax = self.comm._zmin_global_domain + zrel
                    else:
                        qmax = self.select[quantity][1]

                    select_array = np.logical_and(
                        quantity_array < qmax,
                        select_array )

        return( select_array )

    def parse_quantity_dimensions( self, quantity ):
        """
        Parse a given quantity name and retrieve the relevant dimensional units

        Parameters
        ----------
        quantity : string
            The quantity to be parsed
        """
        # Check custom quantities first

        for custom_quantity in self.custom_quantities:
            if quantity == custom_quantity.name:
                return( custom_quantity.dimensions )
        # If no matches, check the common quantities
        if quantity in ('x','y','z'):
            dimensions = unit_dimension_dict['position']
        elif quantity in ('ux','uy','uz'):
            dimensions = unit_dimension_dict['momentum']
        elif quantity == 'gamma':
            dimensions = unit_dimension_dict['gamma']
        elif quantity == 'w':
            dimensions = unit_dimension_dict['weighting']
        elif quantity in ('Ex','Ey','Ez'):
            dimensions = unit_dimension_dict['E']
        elif quantity in ('Bx','By','Bz'):
            dimensions = unit_dimension_dict['B']

        return( dimensions )

    def setup_openpmd_mesh_record( self, dset, dx, xmin ) :
        """
        Sets the attributes that are specific to a mesh record
        Parameter
        ---------
        dset : an h5py.Dataset or h5py.Group object

        dx: list of floats
            The resolution of this diagnostic in each dimension
        xmin: list of floats
            The position of the left edge of the bins in each dimension
        """

        # Axis parameters
        dset.attrs['gridSpacing'] = dx
        dset.attrs["gridGlobalOffset"] = xmin
        dset.attrs['axisLabels'] = self.phase_space
        # Axis dimensions
        gridUnitDimension = np.array([])
        for axis in self.phase_space:
            gridUnitDimension = np.append( gridUnitDimension, self.parse_quantity_dimensions(axis))

        dset.attrs['gridUnitDimension'] = gridUnitDimension
        dset.attrs["gridUnitSI"] = self.Ndims * [1.]

        # Deposited quantity parameters
        dset.attrs['depositedQuantityUnit'] = self.deposit
        dset.attrs['depositedQuantityUnitSI'] = 1.0
        dset.attrs['depositedQuantityUnitDimension'] = self.parse_quantity_dimensions(self.deposit)

        # Generic attributes
        dset.attrs["dataOrder"] = np.bytes_("C")
        dset.attrs["fieldSmoothing"] = np.bytes_("none")

    def get_global_sample_limits(self, root=0):
        """
        Determine the global maxima/minima of the sample from the local values
        then broadcast the result.
        """
        comm = self.comm

        # gather all the mins and maxs
        if self.size > 1:
            allmax = comm.gather_ptcl_array( self.quant_max, comm.size*[self.Ndims],
                                                 comm.size*self.Ndims, root=root)
            allmin = comm.gather_ptcl_array( self.quant_min, comm.size*[self.Ndims],
                                                 comm.size*self.Ndims, root=root)
        else:
            allmax = self.quant_max
            allmin = self.quant_min

        if self.rank==root:
            # reshape the mins/maxs
            allmax = allmax.reshape((comm.size, self.Ndims))
            allmin = allmin.reshape((comm.size, self.Ndims))

            # reduce the mins/maxs ignoring nans
            # if the array is all nans, nanmin/max warn the user and return nan
            # first suppress the warnings,
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.quant_min[:] = np.nanmin(allmin, axis=0)
                self.quant_max[:] = np.nanmax(allmax, axis=0)

            # then replace any remaining nans with dummy limits
            self.quant_max[np.isnan(self.quant_max)] =  0.5
            self.quant_min[np.isnan(self.quant_min)] = -0.5

        # broadcast the now global mins and maxs
        if self.size > 1:
            comm.mpi_comm.Bcast(self.quant_min, root=root)
            comm.mpi_comm.Bcast(self.quant_max, root=root)

    def get_bin_edges(self):
        """
        Parse the provided histogram limits and calculate the specific limits.
        """
        # Initialise the list of bin edges
        bin_edges = []

        # Build the bins edges from the provided limits and global limits
        # and deal with the moving window if required
        if self.edges is None:
            # If no edges are specified, use the global limits (auto-scale)
            bin_edges = [(i,j) for i,j in zip(self.quant_min, self.quant_max)]
        else:
            # otherwise work through the provided bin edges one by one
            for i in range(self.Ndims):
                # start with a pair of dummy limits
                i_min = np.nan
                i_max = np.nan

                # any `None` limits need to be replaced with the global ranges.
                # further, if the phase space being binned is z and comoving
                # limits are in place, adjust the actual limits

                # single `None`: autoscale upper and lower
                if self.edges[i] is None:
                    i_min = self.quant_min[i]
                    i_max = self.quant_max[i]
                else:
                    # pairs of limits treated individually, z is a special case
                    if (self.phase_space[i] == 'z') and (self.move_with_window):
                        if self.edges[i][0] is not None:
                            zmin = self.comm._zmin_global_domain
                            i_min = zmin + (self.edges[i][0]-self.zmin_init)
                        else:
                            i_min = self.quant_min[i]

                        if self.edges[i][1] is not None:
                            zmin = self.comm._zmin_global_domain
                            i_max = zmin + (self.edges[i][1]-self.zmin_init)
                        else:
                            i_max = self.quant_max[i]

                    else:
                        # non-z quantities next
                        if self.edges[i][0] == None:
                            i_min = self.quant_min[i]
                        else:
                            i_min = self.edges[i][0]

                        if self.edges[i][1] == None:
                            i_max = self.quant_max[i]
                        else:
                            i_max = self.edges[i][1]

                bin_edges.append( (i_min, i_max) )

        return bin_edges

    def write_dataset( self, phasespace_grp, species, path,
                       Ntot, select_array ) :
        """
        Create and write a histogram

        Parameters
        ----------
        phasespace_grp : an h5py.Group
            The group where to write the phasespace considered
        species : a fbpic.Particles object
        	The species object to get the particle data from
        path : string
            The relative path where to write the dataset,
            inside the phasespace_grp
        n_rank : list of ints
            A list containing the number of particles to send on each proc
        Ntot : int
        	Contains the global number of particles
        select_array : 1darray of bool
            An array of the same shape as that particle array
            containing True for the particles that satify all
            the rules of self.select
        """
        # Create the datasets and set up their attributes
        if self.rank==0:
            datashape = self.bins
            dtype = 'f8'
            # If the dataset already exists, remove it.
            # (This avoids errors with diags from previous simulations,
            # in case the number of particles is not exactly the same.)
            if path in phasespace_grp:
                del phasespace_grp[path]

            dset = phasespace_grp.create_dataset(path, datashape, dtype=dtype )

        # get the total number of particles to consider on this rank
        Ntot = select_array.sum()

        # reset the min/max arrays
        self.quant_min[:] = np.nan
        self.quant_max[:] = np.nan

        # contruct the sample to be binned
        sample = np.zeros((Ntot, self.Ndims))

        # only do work if there is going to be data at the end of it
        if Ntot > 0:
            for i in range(self.Ndims):

                # get the particle quantity
                quant = self.get_particle_quantity(species, self.phase_space[i],
                                                   select_array)
                # add it to the sample
                sample[:,i] = quant

                # record the min/max of each quantity
                self.quant_min[i] = quant.min()
                self.quant_max[i] = quant.max()

        # get and broadcast the global sample limits
        self.get_global_sample_limits()
        # compute the bin edges
        bin_edges = self.get_bin_edges()

        if Ntot > 0:
            # Get the particle weights
            if self.unweighted:
                weights = np.ones(Ntot)
            else:
                weights = self.get_basic_particle_quantity( species, 'w', select_array)
            # deposit a specified quantity alongside the weights
            if self.deposit != 'w':
                deposit = self.get_particle_quantity(species, self.deposit, select_array)
                weights *= deposit
        else: # dummy zero-length weights
            weights = np.ones(0)

        # Generate the histogram
        # this must still be done for zero-length samples as well
        # we still need to compute the edges to write to disk on the root
        hist, edges = np.histogramdd(sample, bins=self.bins,
                                        density=False, weights=weights,
                                        range=bin_edges)

        # reduce the histograms from each rank onto the root
        if self.rank == 0:
            allhist = np.empty_like(hist)
        else:
            allhist = None
        # sum is the default operation, no need for more imports
        if self.size > 1:
            self.comm.mpi_comm.Reduce(np.ascontiguousarray(hist), allhist, root=0)
        else:
            allhist = hist

        if self.rank==0:
            dset[:] = allhist
            # Fill out the dataset and metadata for the axes
            dx = []
            xmin = []
            for axis in edges:
                dx.append(axis[1]-axis[0])
                xmin.append(axis[0])

            self.setup_openpmd_mesh_record( dset, dx, xmin )

    def get_particle_quantity( self, species, quantity, select_array):
        """
        Retrieve a specified particle quantity
        Parameters
        ----------
        species : a Particles object
        	The species object to get the particle data from

        quantity : string
            The quantity to retrieve
        select_array : 1darray of bool
            An array of the same shape as that particle array
            containing True for the particles that satify all
            the rules of self.select
        """
        # Check the custom quantities first
        for custom_quantity in self.custom_quantities:
            if quantity == custom_quantity.name:

                # Create an empty list of  arguments
                args = []
                # Append the particle arrays to the list
                for arg in custom_quantity.arguments:
                    args.append(self.get_basic_particle_quantity( species, arg, select_array))
                # Pass the arguments to the function
                return( custom_quantity(*args) )
        # If none are found, fall back
        return( self.get_basic_particle_quantity( species, quantity, select_array ) )

    def get_basic_particle_quantity(self, species, quantity, select_array):
        """
        Extract the local array that satisfies select_array
        species : a Particles object
        	The species object to get the particle data from
        quantity : string
            The quantity to be extracted (e.g. 'x', 'uz', 'w')
        select_array : 1darray of bool
            An array of the same shape as that particle array
            containing True for the particles that satify all
            the rules of self.select

        """
        # Extract the quantity
        if quantity == "id":
            quantity_one_proc = species.tracker.id
        elif quantity == "charge":
            quantity_one_proc = constants.e * species.ionizer.ionization_level
        elif quantity == "w":
            quantity_one_proc = species.w
        elif quantity == "gamma":
            quantity_one_proc = 1.0/getattr( species, "inv_gamma" )
        else:
            quantity_one_proc = getattr( species, quantity )

        # Apply the selection
        quantity_one_proc = quantity_one_proc[ select_array ]

        # If this is the momentum, multiply by the proper factor
        # (only for species that have a mass)
        if quantity in ['ux', 'uy', 'uz']:
            if species.m>0:
                scale_factor = species.m * constants.c
                quantity_one_proc *= scale_factor

        # Return the results
        return( quantity_one_proc )