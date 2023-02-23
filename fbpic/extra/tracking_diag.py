import warnings, h5py
import numpy as np

class ParticleTrackingDiagnostic:
    def __init__(self, sim, species, name, tags, diag_period, 
                 stride=1, track=['','u']):
        """
        Define a particle tracking diagnostic. !! One species per diagnostic !!
        
        Automatically collect and record particle data from the given species.
        Requires tags to be turned on for the relevant species.
        
        Parameters
        ----------
        sim: Simulation
            Parent simulation object.
            
        species: Species
            Species to pull data from.

        name: str
            Species name, used to name the output file.
            
        tags: list of int
            List of tags to track. 
        
        diag_period: int
            Interval in steps between writes to disk
            
        stride: int
            Steps between datapoints. Default: 1 
            
        track: list of str
            Quantities to track, any combination of:
              ''  : Position
              'u' : Momentum
              'E' : Electric field
              'B' : Magnetic field
            Default: ['', 'u']
        
        """
        # check tags present
        assert hasattr(species, 'tracker'), 'Target species must have tags turned on!'
        # guarantee buffers will line up
        assert diag_period%stride == 0, 'stride (%i) must be a factor of diag_period (%i)'%(stride, diag_period)
        # ensure no duplicate tags
        assert len(set(tags)) == len(tags), 'Duplicate tags in list, check your input!'
        # warnings...
        if sim.iteration > 0:
            warnings.warn("WARNING: diagnostic initialised mid-simulation, check buffers synchronise properly! -TW\n")
            
        # register the simulation and species objects
        self.sim = sim
        self.comm = sim.comm
        self.species = species
        # sort and register IDs to be tracked
        self.sort = np.argsort(tags)
        self.tags = tags[self.sort]
        self.Ntracks = len(tags)
        # register quantities to be tracked
        self.track = track 
        # register the step at which the diag was initialised
        self.istart = sim.iteration
        # register the stride (steps between datapoints)
        self.stride = stride
        # register the buffer size, and calculate the residual for the first write
        self.diag_period = diag_period
        self.repeating_buffer_size = diag_period // stride
        self.initial_buffer_size = self.repeating_buffer_size - (sim.iteration%diag_period // stride)
        self.buffer_size = self.initial_buffer_size
        # register a time buffer and a dict of buffer object for each particle
        self.tbuffer = np.full(self.repeating_buffer_size, np.nan)
        self.buffer = self.create_buffer()
        # set up the file path
        self.filepath = 'tracking_%s.h5'%name
        # make the empty file on the first proc
        if self.comm.rank == 0:
            with h5py.File(self.filepath, 'w') as f:
                # general information
                f.attrs['dt'] = sim.dt * stride
                # species information
                f.attrs['m'] = self.species.m
                f.attrs['q'] = self.species.q
                # lists of tags, weights, times 
                f.create_dataset( 'tags', data=np.array(self.tags) )
                f.create_dataset( 'w', shape=(len(self.tags),), dtype=np.double )
                f.create_dataset( 't', shape=(0,), dtype=np.double, maxshape=(None,) )  
                # create particle datasets
                for quant in self.track:
                    f.create_dataset( '%sx'%quant, shape=(self.Ntracks,0), maxshape=(self.Ntracks,None) )
                    f.create_dataset( '%sy'%quant, shape=(self.Ntracks,0), maxshape=(self.Ntracks,None) )
                    f.create_dataset( '%sz'%quant, shape=(self.Ntracks,0), maxshape=(self.Ntracks,None) )  
        return

    def create_buffer(self):
        """ 
        Generate buffer arrays for the particle quantities as a dict.
        """
        buffer = {}
        buffer['w'] = np.full(self.Ntracks, np.nan)
        for quant in self.track:
            buffer[quant] = np.full((3,self.Ntracks, self.repeating_buffer_size), np.nan)
        return buffer
    
    def write(self, iteration ):
        """
        This method is called during the main PIC loop
        """
        if iteration % self.diag_period == 0 and iteration > self.istart:
            self.write_data()
        if iteration % self.stride == 0:
            self.gather_data()
        return
    
    def get_i_ptcl(self, ID):
        """ 
        Get the particle's index within the simulation arrays from its ID, 
        return None if particle is missing
        """
        try:
            return np.argwhere( self.species.tracker.id == ID )[0,0]
        except IndexError:
            return None
    
    def get_i_file(self, ID):
        """ 
        Get the particle's index within the buffer/file from its ID
        """
        return np.argwhere( self.tags == ID )[0,0]
    
    
    def gather_data(self): 
        """ Gather particle data to local buffers """
        # get the relative position within the buffer
        i_buff = (self.sim.iteration//self.stride) % self.repeating_buffer_size - self.repeating_buffer_size
        # time is common to all particles
        self.tbuffer[i_buff] = self.sim.time
        # recieve data from the GPU
        if self.species.use_cuda:
                self.species.receive_particles_from_gpu()
        # loop over each ID to be tracked
        for ID in self.tags:
            i_ptcl = self.get_i_ptcl(ID)  
            i_file = self.get_i_file(ID)
            # fill with NaNs if particle is missing
            if i_ptcl is None:
                for quant in self.track:
                    self.buffer[quant][0,i_file,i_buff] = np.nan
                    self.buffer[quant][1,i_file,i_buff] = np.nan
                    self.buffer[quant][2,i_file,i_buff] = np.nan
            # real data if particle is alive
            else:
                self.buffer['w'][i_file] = self.species.w[i_ptcl]
                for quant in self.track:
                    self.buffer[quant][0,i_file,i_buff] = getattr(self.species, '%sx'%quant)[i_ptcl]
                    self.buffer[quant][1,i_file,i_buff] = getattr(self.species, '%sy'%quant)[i_ptcl]
                    self.buffer[quant][2,i_file,i_buff] = getattr(self.species, '%sz'%quant)[i_ptcl]
        # send data back to GPU
        if self.species.use_cuda:
            self.species.send_particles_to_gpu() 
        return
    
    def gather_weights(self, root=0):
        """ gather weights """
        # the local partial list of weights
        local = self.buffer['w']
        # define the global list to be reduced
        glob = np.empty((self.comm.size, self.Ntracks))
        # gather...
        self.comm.mpi_comm.Gather( local, glob, root=root )
        # perform the subsequent reduction only on the root
        if self.comm.rank == root:
            # create the nan mask 
            nans = np.isnan(glob) 
            nans = ~((~nans).any(axis=0))
            # remove nans and reduce the global array to the correct shape
            glob = np.nan_to_num( glob )
            glob = glob.sum(axis=0)
            # remove any suprious axes
            glob = np.squeeze(glob)
            # re-insert the nan values
            glob[nans] = np.nan
            return glob
        else:
            return None
        
        
    def gather_quant(self, quant, n=3, root=0):
        """ gather an n-component quantity from the buffer """
        # make an n-D array on which to gather the buffers
        glob = np.empty((self.comm.size, n, self.Ntracks, self.repeating_buffer_size))
        # define the local copy of the quantity
        local = self.buffer[quant]
        # gather the local buffers into the global array on the root
        self.comm.mpi_comm.Gather( local, glob, root=root )
        # perform the subsequent reduction only on the root
        if self.comm.rank == root:
            # create the nan mask 
            nans = np.isnan(glob) 
            nans = ~((~nans).any(axis=0))
            # remove nans and reduce the global array to the correct shape
            glob = np.nan_to_num( glob )
            glob = glob.sum(axis=0)
            # remove any suprious axes
            glob = np.squeeze(glob)
            # re-insert the nan values
            glob[nans] = np.nan
            return glob
        else:
            return None
        
    def write_data(self, root=0):
        """ Collect data on root, extend file datasets, and append the buffers """
        print('writing')
        # expand the time buffer on the root and update the weights
        weights = self.gather_weights(root=root)
        if self.comm.rank == root:
            with h5py.File(self.filepath, 'a') as f:
                # get current dataset size and expand 
                current_size = f['t'].shape[0]
                f['t'].resize( (current_size + self.buffer_size), axis=0)                    
                f['t'][-self.buffer_size:] = self.tbuffer[-self.buffer_size:]
                # record the weights
                f['w'][:] = weights[:]
                
        # iterate through each quantity
        for quant in self.track:
            
            glob = self.gather_quant( quant, n=3, root=root )
            
            if self.comm.rank == 0:
                with h5py.File(self.filepath, 'a') as f:
                    # resize particle datasets
                    f['%sx'%quant].resize( current_size + self.buffer_size, axis=1)
                    f['%sy'%quant].resize( current_size + self.buffer_size, axis=1)
                    f['%sz'%quant].resize( current_size + self.buffer_size, axis=1)
                    # write buffers
                    for ID in self.tags:
                        # get the file index
                        i_file = self.get_i_file(ID)
                        # tracked data
                        f['%sx'%quant][i_file, -self.buffer_size:] = glob[0,i_file,-self.buffer_size:]               
                        f['%sy'%quant][i_file, -self.buffer_size:] = glob[1,i_file,-self.buffer_size:]  
                        f['%sz'%quant][i_file, -self.buffer_size:] = glob[2,i_file,-self.buffer_size:]  
                        
        # set the buffer size to the repeat period after the first write
        self.buffer_size = self.repeating_buffer_size
        return

