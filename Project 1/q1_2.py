import numpy as np
import scipy
import matplotlib.pyplot as plt


def grad_estimate(x, f, dh):
	'''f=function, dh = step size'''
	gradient = (f(x+dh)-f(x-dh))/float(dh)
	return gradient

	
