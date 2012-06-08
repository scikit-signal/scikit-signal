import numpy as np
from scipy.fftpack import fft, ifft, fftshift

__all__ = ['cwt', 'ccwt', 'icwt', 'SDG', 'Morlet']

class MotherWavelet(object):
    """Class for MotherWavelets.

    Contains methods related to mother wavelets.  Also used to ensure that new
    mother wavelet objects contain the minimum requirements to be used in the
    cwt related functions.

    """

    @staticmethod
    def get_coefs(self):
        """Raise error if method for calculating mother wavelet coefficients is
        missing!

        """

        raise NotImplementedError('get_coefs needs to be implemented for the mother wavelet')

    @staticmethod
    def get_coi_coef(sampf):
        """Raise error if Cone of Influence coefficient is not set in
        subclass wavelet. To follow the convention in the literature, please define your
        COI coef as a function of period, not scale - this will ensure
        compatibility with the scalogram method.

        """

        raise NotImplementedError('coi_coef needs to be implemented in subclass wavelet')

    #add methods for computing cone of influence and mask
    def get_coi(self):
        """Compute cone of influence."""

        y1 =  self.coi_coef * np.arange(0, self.len_signal / 2)
        y2 = -self.coi_coef * np.arange(0, self.len_signal / 2) + y1[-1]
        coi = np.r_[y1, y2]
        self.coi = coi
        return coi

    def get_mask(self):
        """Get mask for cone of influence.

        Sets self.mask as an array of bools for use in np.ma.array('', mask=mask)

        """

        mask = np.ones(self.coefs.shape)
        masks = self.coi_coef * self.scales
        for s in range(0, len(self.scales)):
            if (s != 0) and (int(np.ceil(masks[s])) < mask.shape[1]):
                mask[s,np.ceil(int(masks[s])):-np.ceil(int(masks[s]))] = 0
        self.mask = mask.astype(bool)
        return self.mask

class SDG(MotherWavelet):
    """Class for the SDG MotherWavelet (a subclass of MotherWavelet).

    SDG(self, len_signal = None, pad_to = None, scales = None, sampf = 1,
        normalize = True, fc = 'bandpass')

    Parameters
    ----------
    len_signal : int
        Length of time series to be decomposed.

    pad_to : int
        Pad time series to a total length `pad_to` using zero padding (note,
        the signal will be zero padded automatically during continuous wavelet
        transform if pad_to is set). This is used in the fft function when
        performing the convolution of the wavelet and mother wavelet in Fourier
        space.

    scales : array
        Array of scales used to initialize the mother wavelet.

    sampf : float
        Sample frequency of the time series to be decomposed.

    normalize : bool
        If True, the normalized version of the mother wavelet will be used (i.e.
        the mother wavelet will have unit energy).

    fc : string
        Characteristic frequency - use the 'bandpass' or 'center' frequency of
        the Fourier spectrum of the mother wavelet to relate scale to period
        (default is 'bandpass').

    Returns
    -------
    Returns an instance of the MotherWavelet class which is used in the cwt and
    icwt functions.

    Examples
    --------
    Create instance of SDG mother wavelet, normalized, using 10 scales and the
    center frequency of the Fourier transform as the characteristic frequency.
    Then, perform the continuous wavelet transform and plot the scalogram.

    # x = numpy.arange(0,2*numpy.pi,numpy.pi/8.)
    # data = numpy.sin(x**2)
    # scales = numpy.arange(10)
    #
    # mother_wavelet = SDG(len_signal = len(data), scales = np.arange(10),normalize = True, fc = 'center')
    # wavelet = cwt(data, mother_wavelet)
    # wave_coefs.scalogram()

    Notes
    -----
    None

    References
    ----------
    Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook.  Taylor
      and Francis Group, New York/London. 353 pp.

    """

    def __init__(self,len_signal=None,pad_to=None,scales=None,sampf=1,normalize=True, fc = 'bandpass'):
        """Initilize SDG mother wavelet"""

        self.name='second degree of a Gaussian (mexican hat)'
        self.sampf = sampf
        self.scales = scales
        self.len_signal = len_signal
        self.normalize = normalize

        #set total length of wavelet to account for zero padding
        if pad_to is None:
            self.len_wavelet = len_signal
        else:
            self.len_wavelet = pad_to

        #set admissibility constant
        if normalize:
            self.cg = 4 * np.sqrt(np.pi) / 3.
        else:
            self.cg = np.pi

        #define characteristic frequency
        if fc is 'bandpass':
            self.fc = np.sqrt(5./2.) * self.sampf/(2 * np.pi)
        elif fc is 'center':
            self.fc = np.sqrt(2.) * self.sampf / (2 * np.pi)
        else:
            raise CharacteristicFrequencyError("fc = %s not defined"%(fc,))

        # coi_coef defined under the assumption that period is used, not scale
        self.coi_coef = 2 * np.pi * np.sqrt(2. / 5.) * self.fc # Torrence and
                                                               # Compo 1998

        # compute coefficients for the dilated mother wavelet
        self.coefs = self.get_coefs()

    def get_coefs(self):
        """Calculate the coefficients for the SDG mother wavelet"""

        # Create array containing values used to evaluate the wavelet function
        xi=np.arange(-self.len_wavelet / 2., self.len_wavelet / 2.)

        # find mother wavelet coefficients at each scale
        xsd = -xi * xi / (self.scales[:,np.newaxis] * self.scales[:,np.newaxis])

        if self.normalize is True:
            c=2. / (np.sqrt(3) * np.power(np.pi, 0.25))
        else:
            c=1.

        mw = c * (1. + xsd) * np.exp(xsd / 2.)

        self.coefs = mw

        return mw

class Morlet(MotherWavelet):
    """Class for the Morlet MotherWavelet (a subclass of MotherWavelet).

    Morlet(self, len_signal = None, pad_to = None, scales = None,
           sampf = 1, f0 = 0.849)

    Parameters
    ----------
    len_signal : int
        Length of time series to be decomposed.

    pad_to : int
        Pad time series to a total length `pad_to` using zero padding (note,
        the signal will be zero padded automatically during continuous wavelet
        transform if pad_to is set). This is used in the fft function when
        performing the convolution of the wavelet and mother wavelet in Fourier
        space.

    scales : array
        Array of scales used to initialize the mother wavelet.

    sampf : float
        Sample frequency of the time series to be decomposed.

    f0 : float
        Central frequency of the Morlet mother wavelet.  The Fourier spectrum of
        the Morlet wavelet appears as a Gaussian centered on f0.  f0 defaults
        to a value of 0.849 (the angular frequency would be ~5.336).

    Returns
    -------
    Returns an instance of the MotherWavelet class which is used in the cwt
    and icwt functions.

    Examples
    --------
    Create instance of Morlet mother wavelet using 10 scales, perform the
    continuous wavelet transform, and plot the resulting scalogram.

    # x = numpy.arange(0,2*numpy.pi,numpy.pi/8.)
    # data = numpy.sin(x**2)
    # scales = numpy.arange(10)
    #
    # mother_wavelet = Morlet(len_signal=len(data), scales = np.arange(10))
    # wavelet = cwt(data, mother_wavelet)
    # wave_coefs.scalogram()

    Notes
    -----
    * Morlet wavelet is defined as having unit energy, so the `normalize` flag
      will always be set to True.

    * The Morlet wavelet will always use f0 as it's characteristic frequency, so
      fc is set as f0.

    References
    ----------
    Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook.  Taylor
      and Francis Group, New York/London. 353 pp.

    """

    def __init__(self, len_signal=None, pad_to=None, scales=None, sampf=1,
                 normalize=True, f0=0.849):
        """Initilize Morlet mother wavelet."""

        from scipy.integrate import trapz

        self.sampf = sampf
        self.scales = scales
        self.len_signal = len_signal
        self.normalize = True
        self.name = 'Morlet'

        # set total length of wavelet to account for zero padding
        if pad_to is None:
            self.len_wavelet = len_signal
        else:
            self.len_wavelet = pad_to

        # define characteristic frequency
        self.fc = f0

        # Cone of influence coefficient
        self.coi_coef = 2. * self.sampf / (self.fc + np.sqrt(2. + self.fc**2) *
                        np.sqrt(2)); #Torrence and Compo 1998 (in code)

        # set admissibility constant
        # based on the simplified Morlet wavelet energy spectrum
        # in Addison (2002), eqn (2.39) - should be ok for f0 >0.84
        f = np.arange(0.001, 50, 0.001)
        y = 2. * np.sqrt(np.pi) * np.exp(-np.power((2. * np.pi * f -
            2. * np.pi * self.fc), 2))
        self.cg =  trapz(y[1:] / f[1:]) * (f[1]-f[0])

        # compute coefficients for the dilated mother wavelet
        self.coefs = self.get_coefs()

    def get_coefs(self):
        """Calculate the coefficients for the Morlet mother wavelet."""

        # Create array containing values used to evaluate the wavelet function
        xi=np.arange(-self.len_wavelet / 2., self.len_wavelet / 2.)

        # find mother wavelet coefficients at each scale
        xsd = xi / (self.scales[:,np.newaxis])

        mw = np.power(np.pi,-0.25) * \
                     (np.exp(np.complex(1j) * 2. * np.pi * self.fc * xsd) - \
                     np.exp(-np.power((2. * np.pi * self.fc), 2) / 2.)) *  \
                     np.exp(-np.power(xsd, 2) / 2.)

        self.coefs = mw

        return mw

class Wavelet(object):
    """Class for Wavelet object.

    The Wavelet object holds the wavelet coefficients as well as information on
    how they were obtained.

    """

    def __init__(self, wt, wavelet, weighting_function, signal_dtype, deep_copy=True):
        """Initialization of Wavelet object.

        Parameters
        ----------
        wt : array
            Array of wavelet coefficients.

        wavelet : object
            Mother wavelet object used in the creation of `wt`.

        weighting_function : function
            Function used in the creation of `wt`.

        signal_dtype : dtype
            dtype of signal used in the creation of `wt`.

        deep_copy : bool
            If true (default), the mother wavelet object used in the creation of
            the wavelet object will be fully copied and accessible through
            wavelet.motherwavelet; if false, wavelet.motherwavelet will be a
            reference to the motherwavelet object (that is, if you change the
            mother wavelet object, you will see the changes when accessing the
            mother wavelet through the wavelet object - this is NOT good for
            tracking how the wavelet transform was computed, but setting
            deep_copy to False will save memory).

        Returns
        -------
        Returns an instance of the Wavelet class.

        """

        from copy import deepcopy
        self.coefs = wt[:,0:wavelet.len_signal]

        if wavelet.len_signal !=  wavelet.len_wavelet:
            self._pad_coefs = wt[:,wavelet.len_signal:]
        else:
            self._pad_coefs = None
        if deep_copy:
            self.motherwavelet = deepcopy(wavelet)
        else:
            self.motherwavelet = wavelet

        self.weighting_function = weighting_function
        self._signal_dtype = signal_dtype

    def get_gws(self):
        """Calculate Global Wavelet Spectrum.

        References
        ----------
        Torrence, C., and G. P. Compo, 1998: A Practical Guide to Wavelet
          Analysis.  Bulletin of the American Meteorological Society, 79, 1,
          pp. 61-78.

        """

        gws = self.get_wavelet_var()

        return gws


    def get_wes(self):
        """Calculate Wavelet Energy Spectrum.

        References
        ----------
        Torrence, C., and G. P. Compo, 1998: A Practical Guide to Wavelet
          Analysis.  Bulletin of the American Meteorological Society, 79, 1,
          pp. 61-78.

        """

        from scipy.integrate import trapz

        coef = 1. / (self.motherwavelet.fc * self.motherwavelet.cg)

        wes = coef * trapz(np.power(np.abs(self.coefs), 2), axis = 1);

        return wes

    def get_wps(self):
        """Calculate Wavelet Power Spectrum.

        References
        ----------
        Torrence, C., and G. P. Compo, 1998: A Practical Guide to Wavelet
          Analysis.  Bulletin of the American Meteorological Society, 79, 1,
          pp. 61-78.

        """

        wps =  (1./ self.motherwavelet.len_signal) * self.get_wes()

        return wps

    def get_wavelet_var(self):
        """Calculate Wavelet Variance (a.k.a. the Global Wavelet Spectrum of
        Torrence and Compo (1998)).

        References
        ----------
        Torrence, C., and G. P. Compo, 1998: A Practical Guide to Wavelet
          Analysis.  Bulletin of the American Meteorological Society, 79, 1,
          pp. 61-78.

        """

        coef =  self.motherwavelet.cg * self.motherwavelet.fc

        wvar = (coef / self.motherwavelet.len_signal) * self.get_wes()

        return wvar

def cwt(x, wavelet, weighting_function=lambda x: x**(-0.5), deep_copy=True):
    """Computes the continuous wavelet transform of x using the mother wavelet
    `wavelet`.

    This function computes the continuous wavelet transform of x using an
    instance a mother wavelet object.

    The cwt is defined as:

        T(a,b) = w(a) integral(-inf,inf)(x(t) * psi*{(t-b)/a} dt

    which is a convolution.  In this algorithm, the convolution in the time
    domain is implemented as a multiplication in the Fourier domain.

    Parameters
    ----------
    x : 1D array
        Time series to be transformed by the cwt

    wavelet : Instance of the MotherWavelet class
        Instance of the MotherWavelet class for a particular wavelet family

    weighting_function:  Function used to weight
        Typically w(a) = a^(-0.5) is chosen as it ensures that the
        wavelets at every scale have the same energy.

    deep_copy : bool
        If true (default), the mother wavelet object used in the creation of
        the wavelet object will be fully copied and accessible through
        wavelet.motherwavelet; if false, wavelet.motherwavelet will be a
        reference to the motherwavelet object (that is, if you change the
        mother wavelet object, you will see the changes when accessing the
        mother wavelet through the wavelet object - this is NOT good for
        tracking how the wavelet transform was computed, but setting
        deep_copy to False will save memory).

    Returns
    -------
    Returns an instance of the Wavelet class.  The coefficients of the transform
    can be obtain by the coefs() method (i.e.  wavelet.coefs() )

    Examples
    --------
    Create instance of SDG mother wavelet, normalized, using 10 scales and the
    center frequency of the Fourier transform as the characteristic frequency.
    Then, perform the continuous wavelet transform and plot the scalogram.

    # x = numpy.arange(0,2*numpy.pi,numpy.pi/8.)
    # data = numpy.sin(x**2)
    # scales = numpy.arange(10)
    #
    # mother_wavelet = SDG(len_signal = len(data), scales = np.arange(10), normalize = True, fc = 'center')
    # wavelet = cwt(data, mother_wavelet)
    # wave_coefs.scalogram()

    References
    ----------
    Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook.  Taylor
      and Francis Group, New York/London. 353 pp.

    """

    signal_dtype = x.dtype

    if len(x) < wavelet.len_wavelet:
        n = len(x)
        x = np.resize(x, (wavelet.len_wavelet,))
        x[n:] = 0

    # Transform the signal and mother wavelet into the Fourier domain
    xf=fft(x)
    mwf=fft(wavelet.coefs.conj(), axis=1)

    # Convolve (multiply in Fourier space)
    wt_tmp=ifft(mwf*xf[np.newaxis,:], axis=1)

    # shift output from ifft and multiply by weighting function
    wt = fftshift(wt_tmp,axes=[1]) * weighting_function(wavelet.scales[:, np.newaxis])

    # if mother wavelet and signal are real, only keep real part of transform
    wt=wt.astype(np.lib.common_type(wavelet.coefs, x))

    return Wavelet(wt,wavelet,weighting_function,signal_dtype,deep_copy)

def ccwt(x1, x2, wavelet):
    """Compute the continuous cross-wavelet transform of 'x1' and 'x2' using the
    mother wavelet 'wavelet', which is an instance of the MotherWavelet class.

    Parameters
    ----------
    x1,x2 : 1D array
        Time series used to compute cross-wavelet transform

    wavelet : Instance of the MotherWavelet class
        Instance of the MotherWavelet class for a particular wavelet family

    Returns
    -------
    Returns an instance of the Wavelet class.

    """

    xwt = cwt(x1,wavelet) * np.conjugate(cwt(x2, wavelet))

    return xwt

def icwt(wavelet):
    """Compute the inverse continuous wavelet transform.

    Parameters
    ----------
    wavelet : Instance of the MotherWavelet class
        instance of the MotherWavelet class for a particular wavelet family

    Examples
    --------
    Use the Morlet mother wavelet to perform wavelet transform on 'data', then
    use icwt to compute the inverse wavelet transform to come up with an estimate
    of data ('data2').  Note that data2 is not exactly equal data.

    # import matplotlib.pyplot as plt
    # from scipy.signal import SDG, Morlet, cwt, icwt, fft, ifft
    # import numpy as np
    #
    # x = np.arange(0,2*np.pi,np.pi/64)
    # data = np.sin(8*x)
    # scales=np.arange(0.5,17)
    #
    # mother_wavelet = Morlet(len_signal = len(data), scales = scales)
    # wave_coefs=cwt(data, mother_wavelet)
    # data2 = icwt(wave_coefs)
    #
    # plt.plot(data)
    # plt.plot(data2)
    # plt.show()

    References
    ----------
    Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook.  Taylor
      and Francis Group, New York/London. 353 pp.

    """
    from scipy.integrate import trapz

    # if original wavelet was created using padding, make sure to include
    #   information that is missing after truncation (see self.coefs under __init__
    #   in class Wavelet.
    if wavelet.motherwavelet.len_signal !=  wavelet.motherwavelet.len_wavelet:
        full_wc = np.c_[wavelet.coefs,wavelet._pad_coefs]
    else:
        full_wc = wavelet.coefs

    # get wavelet coefficients and take fft
    wcf = fft(full_wc,axis=1)

    # get mother wavelet coefficients and take fft
    mwf = fft(wavelet.motherwavelet.coefs,axis=1)

    # perform inverse continuous wavelet transform and make sure the result is the same type
    #  (real or complex) as the original data used in the transform
    x = (1. / wavelet.motherwavelet.cg) *
        trapz(fftshift(ifft(wcf * mwf,axis=1),axes=[1]) /
        (wavelet.motherwavelet.scales[:,np.newaxis]**2),
        dx = 1. / wavelet.motherwavelet.sampf, axis=0)


    return x[0:wavelet.motherwavelet.len_signal].astype(wavelet._signal_dtype)