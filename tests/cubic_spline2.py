from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
from deform.utils.utils import *

# theta = 2 * np.pi * np.linspace(0, 1, 5)
# y = np.c_[np.cos(theta), np.sin(theta)]
# cs = CubicSpline(theta, y, bc_type='periodic')
# print("ds/dx={:.1f} ds/dy={:.1f}".format(cs(0, 1)[0], cs(0, 1)[1]))
# #ds/dx=0.0 ds/dy=1.0
# xs = 2 * np.pi * np.linspace(0, 1, 100)
# plt.figure(figsize=(6.5, 4))
# plt.plot(y[:, 0], y[:, 1], 'o', label='data')
# plt.plot(np.cos(xs), np.sin(xs), label='true')
# plt.plot(cs(xs)[:, 0], cs(xs)[:, 1], label='spline')
# plt.axes().set_aspect('equal')
# plt.legend(loc='center')
# plt.show()

xm, ym = generate_initial_points(x=0.2, y=0.5, num_points=30, link_length=0.1)
num_points = 30
#x = np.linspace(min(xm), (max(xm)-min(xm))/(num_points-1), max(xm))
x = np.linspace(min(xm), max(xm), num=num_points)
y = np.c_[xm, ym]
cs = CubicSpline(x, y, bc_type='not-a-knot')


 

print("ds/dx={:.1f} ds/dy={:.1f}".format(cs(0, 1)[0], cs(0, 1)[1]))
#ds/dx=0.0 ds/dy=1.0
xs = np.linspace(xm[0], xm[-1], 100)
plt.xlim(0, 4)
plt.ylim(0, 3)
#plt.plot(y[:, 0], y[:, 1], 'o', label='data')
#plt.plot(np.cos(xs), np.sin(xs), label='true')
#plt.plot(xm, ym, 'bo', label='data')
plt.plot(cs(xs)[:, 0], cs(xs)[:, 1], label='cubic spline')
#plt.axes().set_aspect('equal')
#plt.legend(loc='center')
plt.show()