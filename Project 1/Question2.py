import matplotlib.pyplot as plt
import numpy as np

X = [ 0, 0.1000,    0.2000,    0.3000,    0.4000,  0.5000,    0.6000,    0.7000,    0.8000,    0.9000,    1.0000]
Y = [2.2748,    1.5728,    0.2885,    0.1237,   -0.8100,   -1.5123,   -0.8655,   -0.8766,   -0.6274,   -0.4159,    0.8383]

Z = np.polyfit(X, Y, 3)
p = np.poly1d(Z)
t = np.linspace(0,1,100)

#real = np.cos(np.pi*X)#+1.5*np.cos(2.0*np.pi*X)
plt.plot(X, Y, 'o', t, p(t), '-')

plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Regression')
plt.grid(True)
plt.show()
