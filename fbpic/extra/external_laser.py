import math
from scipy.constants import m_e, e, c, epsilon_0, pi

from fbpic.lpa_utils.external_fields import ExternalField

def add_external_laser(sim, a0, w0, ctau, zf, z_offset=0., lambda0=8e-7,
               theta=0., cep_phase=0., theta_pol=0., n_e=0.,
               gamma_boost=None, t_max=math.inf ):
    """
    Add a laser to the simulation made up of external fields. Only the 
    transverse E and B fields are initialised. The laser is modelled as a 
    Gaussian beam centered on focal position `zf` at an initial offset 
    `z_offset` at time zero. The laser can be given an arbitrary angle of 
    incidence `theta` and polarisation angle `theta_pol`. 
    In all cases, the focal point of the laser occurs at the axis (r = 0).
    
    Note: Lasers in FBPIC are usually initialised with an absolute centroid 
    position `z0`. Here we use `z_offset` instead to distinguish the fact that
    this is not an absolute position,but a relative offset from the focus
    along the path of the laser.  
    
    Parameters
    ----------
    sim: a Simulation object
       The structure that contains the simulation.
       
    a0: float
        Normalised laser amplitude.
    
    w0: float 
        Laser spot size (meters).
        
    ctau: float 
        Laser duration in space (meters).
    
    zf: float
        Laser focal point along z (meters).
        
    z_offset: float, optional
        laser centroid initial offset from zf. Negative values place the
        laser before the focus, positive values place the laser beyond the 
        focus. (meters).
        Default: 0 
        
    lambda0: float, optional
        Laser vacuum wavelength. 
        Default: 800 nm
    
    theta: float, optional
        Laser angle of incidence i.e. rotation in the z-x plane (rad.).
        Default: 0 
        
    theta_pol: float, optional
        Laser polarisation angle in the transverse plane (rad.).
        Default: 0
    
    cep_phase: float, optional
        Carrier-envelope phase (rad.).
        Default: 0
        
    n_e: float, optional
        Reference plasma density (m^-3) for calculating a refractive index.
        Default: 0
        
    gamma_boost: float or None, optional
        Lorentz factor for boosted frame simulations.
        Default: None
        
    t_max: float, optional
        Time limit for initialising the laser. Convenience option to simplify 
        restarts.
        Default: inf
        
    """
    if sim.time > t_max:
        print(f't_max exceeded, skipping external laser ({sim.time} > {t_max})')
        return
    
    k0 = 2*pi/lambda0
    omega0 = 2*pi*c/lambda0
    n_c = omega0**2*m_e*epsilon_0/e**2
    
    N0 = math.sqrt( 1 - n_e/n_c) #refractive index
    vg = c*N0 # group velocity
    zR = N0*pi*w0**2/lambda0 # Rayleigh length
    k = k0*N0 # effective wavenumber
    
    # precompute the trig functions, use conditionals to ensure exact results where we can
    if theta == 0.:
        cos = 1
        sin = 0
    elif theta == pi/2.:
        cos = 0
        sin = 1
    elif theta == pi:
        cos = -1
        sin = 0
    elif theta == 3*pi/2:
        cos = 0
        sin = -1
    else:
        cos = math.cos(theta) 
        sin = math.sin(theta) 
    
    if theta_pol == 0.:
        cos_pol = 1
        sin_pol = 0
    elif theta_pol == pi/2.:
        cos_pol = 0
        sin_pol = 1
    elif theta_pol == pi:
        cos_pol = -1
        sin_pol = 0
    elif theta_pol == 3*pi/2:
        cos_pol = 0
        sin_pol = -1
    else:
        cos_pol = math.cos(theta_pol) 
        sin_pol = math.sin(theta_pol) 
    
    # overall E and B field amplitudes
    E0 = a0*m_e*omega0*c/e
    B0 = a0*m_e*omega0/e
    # define individual component amplitudes
    Ex = E0 * cos * cos_pol 
    Ey = E0       * sin_pol 
    Ez = E0 *-sin * cos_pol  
    Bx = B0 *-cos * sin_pol
    By = B0       * cos_pol
    Bz = B0 * sin * sin_pol
    
    # Encapsulate the laser function to pick up the variables from this scope
    def laser_field( F, xc, y, zc, t , amplitude, length_scale ):
        # rotate the coordinates, set the focal plane zf as the new origin
        # use the precomputed trig functions to save time (probably?)
        z = (zc-zf)*cos + xc*sin
        x = -(zc-zf)*sin + xc*cos
        # now coordinates are in the proper orientation, proceed as normal
        r2 = x**2 + y**2
        w = w0 * math.sqrt(1 + (z/zR)**2) # spot size
        if z == 0.:     # avoid a division by zero at the focus
            curv = 0.
        else:
            R = z*(1 + (zR/z)**2) # radius of curvature
            curv = 0.5*k*r2/R # curvature phase term
        gouy = math.atan(z/zR)  # Gouy phase term
        envelope = math.exp( -r2/w**2 - ((z-z_offset-t*vg)/(ctau))**2 ) # moving envelope
        phase = math.cos( k*z - omega0*t + curv + gouy + cep_phase ) # laser phase
        return( F + amplitude * w0/w * envelope * phase )

    # check component amplitudes to initialise the minimum number of external fields 
    if Ex != 0.:
        sim.external_fields.append(ExternalField( laser_field, 'Ex', Ex, 0., 
                                                  gamma_boost=gamma_boost ))
    if Ey != 0.:
        sim.external_fields.append(ExternalField( laser_field, 'Ey', Ey, 0., 
                                                  gamma_boost=gamma_boost ))
    if Ez != 0.:
        sim.external_fields.append(ExternalField( laser_field, 'Ez', Ez, 0., 
                                                  gamma_boost=gamma_boost ))
    if Bx != 0.:
        sim.external_fields.append(ExternalField( laser_field, 'Bx', Bx, 0., 
                                                  gamma_boost=gamma_boost ))
    if By != 0.:
        sim.external_fields.append(ExternalField( laser_field, 'By', By, 0., 
                                                  gamma_boost=gamma_boost ))
    if Bz != 0.:
        sim.external_fields.append(ExternalField( laser_field, 'Bz', Bz, 0., 
                                                  gamma_boost=gamma_boost ))
    return 