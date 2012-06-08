# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 22:11:07 2011

@author: Nabobalis

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import wavelet

#Pore data
pore = np.load("pore_test.npy")

# Artifical Data
'''
Here we create an artfical sin wave 
'''
x = np.double(np.arange(0,150))
P = 150. # Period
A = np.double(10*np.sin((2.*np.pi*x)/P))

#Pick your data
data = pore

#Set dt (0.25 for pore data)
dt = 0.25
Nlo=0 
Nhi=len(data)


# Current Wavelet
cw=wavelet.Morlet(data,dt,dj=1./100.,padding=True)
scales=cw.scales   
cwt=cw.cwt
pwr=cw.power
scalespec=np.sum(pwr,axis=1)/Nhi
y=cw.periods
coi=cw.getcoi()
x=np.arange(Nlo,Nhi,1.0)/60.


cdict = {
'blue' : ((0., 1, 1), (0.1, 1.0, 1.0), (0.2, 1, 1), (0.34, 1.0, 1.0), (0.65, 0.0, 0.0), (1.0, 0.0, 0.0)),
'green': ((0., 1, 1), (0.1, 0.0, 0.0), (0.2, 0, 0), (0.37, 1.0, 1.0), (0.64, 1.0, 1.0), (1.0, 0.0, 0.0)),
'red'  : ((0., 1, 1), (0.1, 0.0, 0.0), (0.2, 0, 0), (0.66, 1.0, 1.0), (0.89, 1.0, 1.0), (1.0, 0.5, 0.5))
        }

my_cmap = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 2048)


# sets x and y limits
extent = [Nlo/60.,Nhi/60.,np.min(y),30]#np.max(y)
fig = plt.figure()
axwavelet = plt.subplot(111)
im = plt.contourf(x,y,pwr,100, cmap=my_cmap,extent=extent) 
#im = mpl.image.NonUniformImage(axwavelet,extent=extent, interpolation = "nearest",origin="lower",cmap=my_cmap)
#im.set_data(x,y,pwr)
#axwavelet.images.append(im)
#axwavelet.plot(coi,'k')

polys = axwavelet.fill_between(x,coi,np.max(y),visible=False)
axwavelet.add_patch(mpl.patches.PathPatch(polys.get_paths()[0],hatch='x',facecolor=(0,0,0,0)))
axwavelet.axis(extent)
axwavelet.set_title('Wavelet Power Spectrum')
axwavelet.set_xlabel('Time [mins]')
axwavelet.set_ylabel('Period [mins]')
axwavelet.axis()
# create new axes on the right and on the top of the current axes.
divider = make_axes_locatable(axwavelet)
axdata  = divider.append_axes("top", size=1.0, pad=0.6, sharex=axwavelet)
axpower = divider.append_axes("right", size=1.5, pad=0.4, sharey=axwavelet)
axbar  =  divider.append_axes("bottom", size=0.1, pad=0.6)

# Creates a color bar
plt.colorbar(im, cax=axbar, orientation='horizontal')

# Plots orignal data series
axdata.plot(x,data,'k-')
axdata.set_xlim(extent[0:2])
axdata.set_title('What Data')
axdata.set_ylabel('Unit')

# Plots the total power
axpower.plot(scalespec,y,'k-')
axpower.set_title('Global Wavelet')
axpower.set_xlabel('Power')
axpower.xaxis.set_major_locator(mpl.ticker.LinearLocator(3))
axpower.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%1.e"))
axpower.set_ylim(extent[2:4])

plt.show()
