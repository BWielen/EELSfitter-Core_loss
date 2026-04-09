import numpy as np

def a_0():
    """Bohr radius in meters."""	
    return 5.29177210903*10**-11

def eVtoJ():
    """Conversion factor from eV to J."""
    return 1.60217662e-19

def Rydberg():
    """Rydberg energy in eV."""	
    return 13.605693122994

def c():   
    """Speed of light in m/s."""	
    return 3*10**8

def m_e():
    """Electron mass in kg."""	
    return 9.10938356*10**-31

def hbar():
    """Reduced Planck constant in J*s."""
    return 1.0545718*10**-34

def T(E_0):
    """Kinetic energy of the electron in J."""
    return m_e() * v(E_0)**2 / (2*eVtoJ())

def gamma(E_0):
    """Lorentz factor."""
    return 1+E_0*eVtoJ()/(m_e()*c()**2)

def v(E_0):
    """Speed of the electron in m/s."""
    return c() * np.sqrt(1 - 1/gamma(E_0)**2)

def theta_E(E,E_0):
    """characteristic scattering angle."""
    return E*eVtoJ()/(gamma(E_0)*m_e()*v(E_0)**2)