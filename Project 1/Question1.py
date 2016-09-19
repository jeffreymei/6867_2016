import numpy as np
import scipy
from scipy import linalg
import matplotlib.pyplot as plt
import os 
os.chdir("/home/jeffrey/6867_2016/Project 1/P1")
from loadParametersP1 import getData
from loadFittingDataP1 import getData as getFitData

def df(x):
	y = 4 * x**3 - 9 * x**2
	return y

def df2(x, n, sigma, u):
	f= -1./(np.sqrt((2.*np.pi)**n*linalg.det(sigma)))*np.exp(-0.5*(x-u).T.dot(linalg.inv(sigma)).dot((x-u)))
 	result = -f*linalg.inv(sigma).dot((x-u))
	return result


def df3 (x,A,b):
 	f= 0.5*x.T.dot(A).dot(x)-x.T.dot(b)
 	result =A.dot(x)-b
 	return result

n = 1

x_old = np.array([0.,0.]) # The value does not matter as long as abs(x_new - x_old) > precision
x_new = np.array([8,12]) # The algorithm starts at x=6
gamma = .01 # gamma should be >1000 for gaussian, <0.1 for quadBowl
precision = 1e-5

(gaussMean,gaussCov,quadBowlA,quadBowlb)  = getData()
count=0


while np.sqrt(sum((x_new-x_old)**2)) > precision and count<10000:
	x_old = x_new
	x_new = x_new -gamma * df3(x_old, quadBowlA, quadBowlb)
print("The local minimum occurs at "+str(x_new))
# while np.sqrt(sum((x_new-x_old)**2)) > precision and count<10000:
# 	x_old = x_new
# 	x_new = x_new - gamma* df2(x_old, 1, gaussCov, gaussMean)
# 	count+=1
# print("The local minimum occurs at "+str(x_new))

sys.exit()


steps = 0
