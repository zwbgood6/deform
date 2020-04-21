import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np

x = np.array([1,2,3,4,5,0,7,6,8,9])
y = np.exp(-x/3.0)
f = interpolate.interp1d(x, y, kind='cubic')

xnew = np.arange(0, 9, 0.1)
ynew = f(xnew)   # use interpolation function returned by `interp1d`
plt.plot(x, y, 'o', xnew, ynew, '-')
plt.show()