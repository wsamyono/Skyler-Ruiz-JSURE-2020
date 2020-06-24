#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:41:59 2020

@author: skyler
"""


import numpy as np
from scipy.integrate import odeint
import scipy.optimize 
import matplotlib.pyplot as plt

# function that returns dy/dt
def model(ym,tm):
    k = 0.8
    dydt = -k * ym
    return dydt

# initial condition
y0 = 600000

# time points
tm = np.linspace(0, 5)

# solve ODE
ym = odeint(model,y0,tm)

td = [0, 1, 2, 3, 4, 5]
yd = [50000, 17500, 135000, 350000, 525000, 172500]
plt.plot(td,yd)

# plot results
plt.plot(tm,ym)
plt.xlabel('Days')
plt.ylabel('Cell Growth/ml')
plt.title('Growth Curve')
plt.show()