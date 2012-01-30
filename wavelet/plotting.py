# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 22:11:07 2011

@author: Nabobalis

"""

import idlsave
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import wavelet
import old_wavelet

# IDL Data
data = idlsave.read('results.save')
IDL = data.smooth_area_1
# Artifical Data
x = np.double(np.arange(0,150))
P = 125. # Period
y = np.double(10*np.sin((2.*np.pi*x)/P))
A= IDL 
dt = 0.25
scaling="linear" #or "log"
Ns=len(A)
Nlo=0 
Nhi=Ns

# Current Wavelet
cw=wavelet.Morlet(A,dt,dj=1./100.,scaling=scaling,padding=True)
scales=cw.getscales()     
cwt=cw.getdata()
pwr=cw.getpower()
scalespec=np.sum(pwr,axis=1)/Ns
y=cw.getperiods()
print np.shape(y)
x=np.arange(Nlo,Nhi,1.0) 

# sets x and y limits
extent = [Nlo,Nhi,np.min(y),np.max(y)]
plt.ion()
fig = plt.figure()
axwavelet = plt.subplot(111)
im = mpl.image.NonUniformImage(axwavelet,extent=extent, interpolation = "nearest",origin="lower")
im.set_data(x,y,pwr)
axwavelet.images.append(im)
axwavelet.axis(extent)
axwavelet.set_title('Wavelet Power Spectrum')
axwavelet.set_xlabel('Time [mins]')
axwavelet.set_ylabel('Period [mins]')

# create new axes on the right and on the top of the current axes.
divider = make_axes_locatable(axwavelet)
axdata  = divider.append_axes("top", size=1.0, pad=0.6, sharex=axwavelet)
axpower = divider.append_axes("right", size=1.5, pad=0.4, sharey=axwavelet)
axbar  =  divider.append_axes("bottom", size=0.1, pad=0.6)

# Creates a color bar
plt.colorbar(im, cax=axbar, orientation='horizontal')

# Plots orignal data series
axdata.plot(x,A,'b-')
axdata.set_xlim(extent[0:2])
axdata.set_title('What Data')
axdata.set_ylabel('Unit')

# Plots the total power
axpower.plot(scalespec,y,'b-')
axpower.set_title('Global Wavelet')
axpower.set_xlabel('Power')
axpower.xaxis.set_major_locator(mpl.ticker.LinearLocator(3))
axpower.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%1.e"))
axpower.set_ylim(extent[2:4])

#plt.show()