from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
from deform.utils.utils import *

xm, ym = generate_initial_points(x=0.2, y=0.5, num_points=30, link_length=0.1)


m = GEKKO()
m.x = m.Param(value=np.linspace(xm[0], xm[-1]))
m.y = m.Var()
m.options.IMODE = 2
m.cspline(m.x, m.y, xm, ym)
m.solve(disp=False, remore=False)

plt.xlim(0, 4)
plt.ylim(0, 3)
plt.plot(xm, ym, 'bo', label='data')
plt.plot(m.x.value,m.y.value,'r--',label='cubic spline')
plt.show()