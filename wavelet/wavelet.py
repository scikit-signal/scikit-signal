"""
A module which implements the continuous wavelet transform

Implemented Wavlet Familes
--------------------------

===========     ===============================================================
Name            Description
===========     ===============================================================
Morlet      
MorletReal   
Paul            Paul (Default order m=2)
DOG             Deriviative of Gaussian
MexicanHat      Order m=2 as above
Haar            Unnormalised version of continuous Haar transform
HaarW           Normalised Haar
===========     ===============================================================



References
----------
- [1] C. Torrance and G.P Compo. A Practical Guide to Wavelet Analysis. 
    Bulletin of the American Meteorological Society, Vol 79 No 1 61-78, 
    January 1998

- [2] I. De Moortel, S.A Munday and A. W. Hood. Wavelet Analyisis, The effect 
    of varying basic wavelet parameters. Solar Physics Vol 222. 203-228 May 04


Notes
-----
This class is based upon the code here: 
    http://www.phy.uct.ac.za/courses/python/examples/moreexamples.html

No permission obtained, credits to the original authors for
    - Cwt Class
    - Orginal 8 Familes

Modified as of Jan 2012 by Stuart Mumford and Nabil Freij

Modifications Implemented
-------------------------
    - Data Padding (As in [1]) to make data up to 2^n
    - Arbitary input timestep
    - Arbitary input smallest scale (default 2*dt)
    - Cone of Influence [1]
    - Removed DOG1 DOG4 class in favour of one m order DOG class
    - Made MexicanHat a subclass of DOG m=2
    - Dropped Log scaling in favour of doing it in the plotting routine
        - Removed notes input as only used for log scaling

Modifications Planned
---------------------
    - Arbitary omega0 (order for Morlet)
    - Significance Contouring
"""

import numpy as np

class Cwt:
    """
    Base class for continuous wavelet transforms. 
    - Implements cwt via the Fourier transform
    - Used by subclass which provides the method wf(self,s_omega)
    - wf is the Fourier transform of the wavelet function.
    Returns an instance.
    
    To be used via subclasses with implemented wf(self,s_omega)
    """
    fourierwl=1.00

    def __init__(self, data, dt, smallestscale = None, dj = 0.125,
                 order=2, padding=True):
        """
        Continuous wavelet transform of data

        data:    data in array to transform
        dt:      data timestep
        largestscale: largest scale as inverse fraction of length
                 of data array
                 scale = len(data)/largestscale
                 smallest scale should be >= 2 for meaningful data
        order:   Order of wavelet basis function for some families
        padding: Set to false to prevent padding up to nearest N = 2^x
        """
        #Set default scale
        smallestscale = smallestscale or 2*dt
        self.order = order
        self.dt = dt
        #make sure data is ndarray
        data = np.array(data)
        if len(data.shape) > 1:
            raise ValueError("Data should be a 1D time series")
        self.ndata = data.shape[0]
        #Pad data up to nearset 2^N
        if padding:
            nearestN = 2**int(np.ceil(np.log(self.ndata)/np.log(2)))
            if nearestN == self.ndata:
                pass
            else:              
                newdata = np.zeros([nearestN])
                newdata[:self.ndata] = data
                data = newdata
                #Just to make sure
                del newdata
        else:
            nearestN = self.ndata
            
        self._setscales(smallestscale,  dj)
        self.cwt = np.zeros([self.nscale,self.ndata], dtype=np.complex64)
        omega = np.array(range(0, nearestN/2) + range(-nearestN/2, 0))*(2.0*np.pi/(nearestN*self.dt))
        datahat = np.fft.fft(data)	        
        
        # loop over scales and compute wvelet coeffiecients at each scale
        # using the fft to do the convolution
        for scaleindex, self.currentscale in enumerate(self.scales):
            s_omega = omega*self.currentscale
            psihat = self.wf(s_omega)
            psihat = psihat *  np.sqrt((2.0*np.pi*self.currentscale)/self.dt)
            convhat = psihat * datahat * np.exp(1j * omega * nearestN * self.dt)
            W    = np.fft.ifft(convhat)
            self.cwt[scaleindex, :self.ndata] = W[:self.ndata]
        return
    
    def _setscales(self, smallestscale, dj):
        """
        Completely re-written for smallest scale.
        """
        J = int(np.log2((self.ndata * self.dt )/ smallestscale) / dj) + 1
        self.nscale = int(J)
        self.scales = np.zeros([J])
        for j in xrange(int(J)):
            self.scales[j] = smallestscale * 2.**(j*dj)
        
    def getdata(self):
        """
        returns wavelet coefficient array
        """
        return self.cwt
        
    def getcoefficients(self):
        """
        Return raw wavelet coefficients
        """        
        return self.cwt
        
    def getpower(self):
        """
        returns square of wavelet coefficient array
        """
        return (self.cwt* np.conjugate(self.cwt)).real
        
    def getscales(self):
        """
        returns array containing scales used in transform
        """
        return self.scales
        
    def getperiods(self):
        """
        returns array containing scales used in transform
        """
        return self.scales * self.fourierwl
        
    def getcoi(self):
        """
        return number of scales
        """
        coi_aa = np.array(range(0, (self.ndata+1)/2) + range(self.ndata/2, 0, -1)) *self.dt
        coi = self.coi * coi_aa 
        return coi

# wavelet classes    
class Morlet(Cwt):
    """
    Morlet wavelet
    """
    _omega0=6.0
    fourierwl=(4.* np.pi)/(_omega0+ np.sqrt(2.0+_omega0**2.))
    coi = fourierwl/np.sqrt(2)
    
    def wf(self, s_omega):
        H = np.ones(len(s_omega))
        for i in range(len(s_omega)):
            if s_omega[i] < 0.0: H[i]=0.0

        xhat =  np.pi**(-0.25) * np.exp(-((s_omega-self._omega0)**2.)/2.) * H
        return xhat

class MorletReal(Cwt):
    """
    Real Morlet wavelet
    """
    _omega0=5.0
    fourierwl=4* np.pi/(_omega0+ np.sqrt(2.0+_omega0**2))
    def wf(self, s_omega):
        H= np.ones(len(s_omega))
        for i in range(len(s_omega)):
            if s_omega[i] < 0.0:
                H[i]=0.0
        # !!!! note : was s_omega/8 before 17/6/03
        xhat=np.pi**(-0.25)*(   np.exp(-(s_omega-self._omega0)**2/2.0) + 
                                np.exp(-(s_omega+self._omega0)**2/2.0) -
                                np.exp(-(self._omega0)**2/2.0) +
                                np.exp(-(self._omega0)**2/2.0)  )
        return xhat

class Paul(Cwt):
    """
    Paul order m wavelet
    """
    def wf(self, s_omega):
        Cwt.fourierwl = 4.* np.pi/(2.*self.order+1.)
        m = self.order
        n = len(s_omega)
        normfactor = float(m)
        for i in range(1,2*m):
            normfactor = normfactor*i
        normfactor = 2.0**m/ np.sqrt(normfactor)
        xhat = np.zeros(n)
        xhat[0:n/2] = normfactor*s_omega[0:n/2]**m* np.exp(-s_omega[0:n/2])
        #return 0.11268723*s_omega**2*exp(-s_omega)*H
        return xhat

class DOG(Cwt):
    """
    Derivative Gaussian wavelet of order m
    but reconstruction seems to work best with +!
    """
    def wf(self, s_omega):
        try:
            from scipy.special import gamma
        except ImportError:
            raise ImportError("IMPORT ERROR: Requires scipy gamma function")
        Cwt.fourierwl=2* np.pi/ np.sqrt(self.order + 0.5)
        m = self.order
        dog = 1.0j**m*s_omega**m* np.exp(-s_omega**2./2.)/ np.sqrt(gamma(self.order + 0.5))
        return dog


class MexicanHat(DOG):
    """
    2nd Derivative Gaussian (mexican hat) wavelet.
    Retained for calling only.
    """
    def __init__(self, data, dt, smallestscale=None, dj=0.125, padding=True):
        self.order = 2
        DOG.__init__(self, data, dt, smallestscale, dj, padding)
    
    
class Haar(Cwt):
    """
    Continuous version of Haar wavelet
    """
    #    note: not orthogonal!
    #    note: s_omega/4 matches Lecroix scale defn.
    #          s_omega/2 matches orthogonal Haar
    # 2/8/05 constants adjusted to match artem eim

    fourierwl = 1.0#1.83129  #2.0
    def wf(self, s_omega):
        haar = np.zeros(len(s_omega),dtype=np.complex64)
        om = s_omega[:]/self.currentscale
        om[0] = 1.0  #prevent divide error
        #haar.imag=4.0*sin(s_omega/2)**2/om
        haar.imag = 4.0* np.sin(s_omega/4.)**2./om
        return haar

class HaarW(Cwt):
    """
    Continuous version of Haar wavelet (norm)
    """
    #    note: not orthogonal!
    #    note: s_omega/4 matches Lecroix scale defn.
    #          s_omega/2 matches orthogonal Haar
    # normalised to unit power

    fourierwl = 1.83129 * 1.2  #2.0
    def wf(self, s_omega):
        haar = np.zeros(len(s_omega),dtype=np.complex64)
        om = s_omega[:]#/self.currentscale
        om[0] = 1.0  #prevent divide error
        #haar.imag=4.0*sin(s_omega/2)**2/om
        haar.imag = 4.* np.sin(s_omega/2.)**2./om
        return haar