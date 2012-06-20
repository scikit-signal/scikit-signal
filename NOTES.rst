Currently two versions of a cwt exist in this repo.

One is Nabil and myselfs attempt based on: http://www.phy.uct.ac.za/courses/python/examples/moreexamples.html
The other is from Sean Arms work on developing CWT for scipy.

Personally my target for this cwt is to get it into SciPy asap, currently I am working on it in here to remove some clutter.

=====================================================================

The two versions of this code need to be compressed into one, Sean's CWT is better documented etc. than ours, so I think we will base on that.

I am not convinced by the layout of the code, mainly the calling a function in the module(cwt, icwt etc.) 
and returning an instance of a class (Wavelet) I think it would be better to have a class which is subclassed for different functions 
but after all what do I know about OOP?!

which brings us to a TODO list:

1) Fathom out WTF an admissability coefficient is
2) Make a damn decision about code layout and get on with it
3) Implement other wavelet families from out CWT version.
4) Write a tutorial documentation
5) Party.

Notes of the Nab (The one and only):

So, called cwt like this:

base = np.linspace(0,5000,5000)
data =6* np.sin(2*np.pi*base/150) + 3*np.cos(2*np.pi*base/300)

scales = np.arange(150)
mother_wavelet = cwt.Morlet(len_signal=len(data), scales = scales)
wavelet = cwt.cwt(data, mother_wavelet)
wavelet.scalogram()

First thing, want to make it this way: sp.cwt(.......) in the future
Second, the scale number should be created automatically. Right now, the scale number seems to be generated manually, which is fine for a discrete wavelet and we can make this class do both types of waveelet.
Thirdly, I'd like to remove the second line that defines the mother_wavelet. It would be nice to be able to call a name of the mother and have it be generated when cwt is called instead.