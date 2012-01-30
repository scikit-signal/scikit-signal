"""
A module which implements the continuous wavelet transform

Implemented Wavlet Familes
--------------------------

===========     ===============================================================
Name            Description
===========     ===============================================================
Morlet      
MorletReal  
MexicanHat  
Paul            Paul (Default order m=2)
DOG1            1st Derivative Of Gaussian
DOG4            4th Derivative Of Gaussian
Haar            Unnormalised version of continuous Haar transform
HaarW           Normalised Haar
===========     ===============================================================



References
----------
- [1] C. Torrance and G.P Compo. A Practical Guide to Wavelet Analysis. Bulletin of the American Meteorological Society, Vol 79 No 1 61-78, January 1998

Notes
-----
This class is based upon the code here: http://www.phy.uct.ac.za/courses/python/examples/moreexamples.html

No permission obtained, credits to the original authors for
    - Cwt Class
    - Orginal 8 Familes

Modified as of Jan 2012 by Stuart Mumford and Nabil Freij

Modifications Implemented
-------------------------
    - Data Padding (As in [1]) to make data up to 2^n
    - Arbitary input timestep
    - Arbitary input smallest scale (default 2*dt)

Modifications Planned
---------------------
    - Cone of Influence [1]
    - Significance Contouring
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

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
                 notes=0, order=2, scaling='linear', padding=True):
        """
        Continuous wavelet transform of data

        data:    data in array to transform, length must be power of 2
        dt:      data timestep
        notes:   number of scale intervals per octave
        largestscale: largest scale as inverse fraction of length
                 of data array
                 scale = len(data)/largestscale
                 smallest scale should be >= 2 for meaningful data
        order:   Order of wavelet basis function for some families
        scaling: Linear or log
        padding: Set to false to prevent padding up to nearest 2^N
        """
        #Set default scale
        smallestscale = smallestscale or 2*dt
        self.order = order
        #make sure data is ndarray
        data = np.array(data)
        if len(data.shape) > 1:
            raise ValueError("Data should be a 1D time series")
        ndata = data.shape[0]
        #Pad data up to nearset 2^N
        if padding:
            nearestN = 2**int(np.ceil(np.log(ndata)/np.log(2)))
            if nearestN == ndata:
                pass
            else:              
                newdata = np.zeros([nearestN])
                newdata[:ndata] = data
                data = newdata
                #Just to make sure
                del newdata
        else:
            nearestN = ndata
            
        self._setscales(ndata,dt,smallestscale,dj,notes,scaling)
        self.cwt = np.zeros([self.nscale,ndata], dtype=np.complex64)
        omega = np.array(range(0,nearestN/2)+range(-nearestN/2,0))*(2.0*np.pi/(nearestN*dt))
        #datahat = np.fft.fft(data)
        datahat = sp.fft(data)								
        
        # loop over scales and compute wvelet coeffiecients at each scale
        # using the fft to do the convolution
        for scaleindex,self.currentscale in enumerate(self.scales):
            s_omega = omega*self.currentscale
            psihat = self.wf(s_omega)
            psihat = psihat *  np.sqrt((2.0*np.pi*self.currentscale)/dt)
            convhat = psihat * datahat * np.exp(1j * omega * nearestN * dt)
            #W    = np.fft.ifft(convhat)
            W    = sp.ifft(convhat)
            self.cwt[scaleindex,:ndata] = W[:ndata]
        return
    
    def _setscales(self,ndata,dt,smallestscale,dj,notes,scaling):
        """
        if notes non-zero, returns a log scale based on notes per ocave
        else a linear scale
        (25/07/08): fix notes!=0 case so smallest scale at [0]
        """
        """
        if scaling=="log":
            if notes<=0: 
                notes=1
            # adjust nscale so smallest scale is 2 
            noctave=self._log2( (ndata * dt)/smallestscale)
            self.nscale=notes*noctave
            self.scales=np.zeros(self.nscale,float)
            
            for j in range(self.nscale):
                self.scales[j] = ndata/(self.scale*(2.0**(float(self.nscale-1-j)/notes)))
        """
        if scaling == "log":
            raise NotImplementedError("Go fuck yourself")
            
        elif scaling=="linear":
            J = int(np.log2((ndata * dt )/ smallestscale) / dj) + 1
            self.nscale = int(J)
            self.scales = np.zeros([J])
            for j in xrange(int(J)):
                self.scales[j] = smallestscale * 2.**(j*dj)
        else: 
            raise ValueError("scaling must be linear or log")
        return
    
    def getdata(self):
        """
        returns wavelet coefficient array
        """
        return self.cwt
        
    def getcoefficients(self):
        
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
        
    def getnscale(self):
        """
        return number of scales
        """
        return self.nscale
        
    def getcoi(self):
        """
        return number of scales
        """
        self.coi = self.fourierwl * dt * np.array(range((nearestN+1)/2)+range(-nearestN/2,0))
        return self.coi

# wavelet classes    
class Morlet(Cwt):
    """
    Morlet wavelet
    """
    _omega0=6.0
    fourierwl=(4.* np.pi)/(_omega0+ np.sqrt(2.0+_omega0**2.))
    
    def wf(self, s_omega):
        H = np.ones(len(s_omega))
        for i in range(len(s_omega)):
            if s_omega[i] < 0.0: H[i]=0.0
        # !!!! note : was s_omega/8 before 17/6/03
        xhat = (np.pi**(-0.25))*(np.exp(-((s_omega-self._omega0)**2.0)/2.0))*H
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
        Cwt.fourierwl=4* np.pi/(2.*self.order+1.)
        m=self.order
        n=len(s_omega)
        normfactor=float(m)
        for i in range(1,2*m):
            normfactor=normfactor*i
        normfactor=2.0**m/ np.sqrt(normfactor)
        xhat= np.zeros(n)
        xhat[0:n/2]=normfactor*s_omega[0:n/2]**m* np.exp(-s_omega[0:n/2])
        #return 0.11268723*s_omega**2*exp(-s_omega)*H
        return xhat

class MexicanHat(Cwt):
    """
    2nd Derivative Gaussian (mexican hat) wavelet
    """
    fourierwl=2.0* np.pi/ np.sqrt(2.5)
    def wf(self, s_omega):
        # should this number be 1/sqrt(3/4) (no pi)?
        #s_omega = s_omega/self.fourierwl
        #print max(s_omega)
        a=s_omega**2
        b=s_omega**2/2
        return a* np.exp(-b)/1.1529702
        #return s_omega**2*exp(-s_omega**2/2.0)/1.1529702

class DOG4(Cwt):
    """
    4th Derivative Gaussian wavelet
    see also T&C errata for - sign
    but reconstruction seems to work best with +!
    """
    fourierwl=2.0* np.pi/ np.sqrt(4.5)
    def wf(self, s_omega):
        return s_omega**4* np.exp(-s_omega**2/2.0)/3.4105319

class DOG1(Cwt):
    """
    1st Derivative Gaussian wavelet
    but reconstruction seems to work best with +!
    """
    fourierwl=2.0* np.pi/ np.sqrt(1.5)
    def wf(self, s_omega):
        dog1= np.zeros(len(s_omega),dtype=np.complex64)
        dog1.imag=s_omega* np.exp(-s_omega**2/2.0)/np.sqrt(np.pi)
        return dog1

class DOG(Cwt):
    """
    Derivative Gaussian wavelet of order m
    but reconstruction seems to work best with +!
    """
    def wf(self, s_omega):
        try:
            from scipy.special import gamma
        except ImportError:
            print "Requires scipy gamma function"
            raise ImportError
        Cwt.fourierwl=2* np.pi/ np.sqrt(self.order+0.5)
        m=self.order
        dog=1.0J**m*s_omega**m* np.exp(-s_omega**2/2)/ np.sqrt(gamma(self.order+0.5))
        return dog

class Haar(Cwt):
    """
    Continuous version of Haar wavelet
    """
    #    note: not orthogonal!
    #    note: s_omega/4 matches Lecroix scale defn.
    #          s_omega/2 matches orthogonal Haar
    # 2/8/05 constants adjusted to match artem eim

    fourierwl=1.0#1.83129  #2.0
    def wf(self, s_omega):
        haar= np.zeros(len(s_omega),dtype=np.complex64)
        om = s_omega[:]/self.currentscale
        om[0]=1.0  #prevent divide error
        #haar.imag=4.0*sin(s_omega/2)**2/om
        haar.imag=4.0* np.sin(s_omega/4)**2/om
        return haar

class HaarW(Cwt):
    """
    Continuous version of Haar wavelet (norm)
    """
    #    note: not orthogonal!
    #    note: s_omega/4 matches Lecroix scale defn.
    #          s_omega/2 matches orthogonal Haar
    # normalised to unit power

    fourierwl=1.83129*1.2  #2.0
    def wf(self, s_omega):
        haar= np.zeros(len(s_omega),dtype=np.complex64)
        om = s_omega[:]#/self.currentscale
        om[0]=1.0  #prevent divide error
        #haar.imag=4.0*sin(s_omega/2)**2/om
        haar.imag=4.0* np.sin(s_omega/2)**2/om
        return haar