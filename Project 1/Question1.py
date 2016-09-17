import numpy as np
import scipy
from scipy import linalg
import matplotlib.pyplot as plt

x_old = np.array([0.,0.]) # The value does not matter as long as abs(x_new - x_old) > precision
x_new = np.array([3.1,3.1]) # The algorithm starts at x=6
gamma = 0.01 # step size
precision = 0.00000001


def df(x):
	y = 4 * x**3 - 9 * x**2
	return y

def df2(x, n, sigma, u):
	
	f= -1./(np.sqrt((2.*np.pi)**n*linalg.det(sigma)))*np.exp(-0.5*(x-u).T.dot(linalg.inv(sigma)).dot((x-u)))
 
 	result = -f*linalg.inv(sigma).dot((x-u))

	return f, result

def df3 (x,A,b):
 	f= 0.5*x.T*A*x-x.T*b
 	result =A*x-b

 	return result

sigma = np.eye(2)
n = 1
u = np.array([3.2,2])

#sys.exit()
while abs(x_new - x_old)[0] > precision*np.ones(2)[0]:
	x_old = x_new
	x_new += -gamma * df2(x_old, n=n, sigma=sigma, u=u)[1]

print("The local minimum occurs at "+str(x_new))


sys.exit()
plt.plot(xs, df(xs))
plt.show()
