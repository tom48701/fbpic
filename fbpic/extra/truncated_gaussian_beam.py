import warnings
import numpy as np
from scipy.constants import c
from scipy.stats import truncnorm
from scipy.special import erf

from fbpic.lpa_utils.bunch import add_particle_bunch_from_arrays

def add_particle_bunch_truncated_gaussian(sim, q, m, sig_r, sig_z, n_emit, gamma0,
                                sig_gamma, n_physical_particles,
                                n_macroparticles, tf=0., zf=0., boost=None,
                                save_beam=None, z_injection_plane=None,
                                initialize_self_field=True,
                                zmin=-np.inf, zmax=np.inf):
    """
    Introduce a truncated Gaussian particle bunch in the simulation,
    along with its space charge field.
    
    The beam distribution is truncated at `zmin` and `zmax`, thus allowing the 
    user to guarantee that exactly `n_macroparticles` macroparticles will be
    initialised between `zmin` and `zmax` even if the full-sized bunch is 
    larger than the box.
    `n_physical_particles` should still correspond to the number of
    particles in the _full_ beam!

    Parameters
    ----------
    See `add_particle_bunch_gaussian` for most options.
       
    zmin: float, optional
        Minimum point at which to truncate the beam (Default: -inf)
        
    zmax: float, optional
        Maximum point at which to truncate the beam (Default: inf)
    """
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
    # Get Gaussian particle distribution in x,y
    x = sig_r * np.random.normal(0., 1., n_macroparticles)
    y = sig_r * np.random.normal(0., 1., n_macroparticles)
    # create the truncated particle distribution in z
    a = (zmin - zf) / sig_z
    b = (zmax - zf) / sig_z
    z = truncnorm.rvs(a, b, loc=zf, scale=sig_z, size=n_macroparticles)

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
    
    # Get weight of each particle, adjusted for the truncation
    w_adj = .5*(erf((zmax-zf)/np.sqrt(2)/sig_z) - erf((zmin-zf)/np.sqrt(2)/sig_z))
    w = w_adj * n_physical_particles / N_new * np.ones_like(x)
    
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

    # Add the electrons to the simulation
    ptcl_bunch = add_particle_bunch_from_arrays(sim, q, m, x, y, z, ux, uy, uz,
                    w, boost=boost, z_injection_plane=z_injection_plane,
                    initialize_self_field=initialize_self_field)
    return ptcl_bunch


if __name__ == '__main__':
    
    kp = 4978.744596358355
    
    import matplotlib.pyplot as plt
    
    n_macroparticles = 1000000
    zf = 0.
    sig_z = 12e-2
    
    zmin = -0.2
    zmax = 0.2
    
    
    full = np.sqrt(2*np.pi)*sig_z
    trunc = np.sqrt(np.pi/2)*sig_z * (erf((zmax-zf)/np.sqrt(2)/sig_z) - erf((zmin-zf)/np.sqrt(2)/sig_z))
    
    frac = (erf((zmax-zf)/np.sqrt(2)/sig_z) - erf((zmin-zf)/np.sqrt(2)/sig_z))/2. #trunc/full
    
    print( frac )

    a = (zmin - zf) / sig_z
    b = (zmax - zf) / sig_z
    z_trunc = truncnorm.rvs(a, b, loc=zf, scale=sig_z, size=n_macroparticles)
    
    z_norm = zf + sig_z * np.random.normal(0., 1., n_macroparticles)
    
    #plt.hist( z_trunc, bins=100, density=True) 
    #plt.hist( z_norm, bins=100, alpha=0.5, density=True) 
    
    z = np.linspace(-.5, .5, 10000)
    
    plt.plot( z, frac*truncnorm.pdf(z, a,b, loc=zf, scale=sig_z) )
    plt.plot( z, truncnorm.pdf(z, -np.inf,np.inf, loc=zf, scale=sig_z) )
    
    #plt.xlim(-49/kp, 1/kp)
    