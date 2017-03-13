

from numpy import genfromtxt, zeros
# read the first 4 columns
data = genfromtxt('iris.csv',delimiter=',',usecols=(0,1,2,3))
# read the fifth column
target = genfromtxt('iris.csv',delimiter=',',usecols=(4),dtype=str)


from pylab import plot , show
plot(data[target=='setosa',0],data[target=='setosa',2],'bo')
plot(data[target=='versicolor',0],data[target=='versicolor',2],'ro')
plot(data[target=='virginica',0],data[target=='virginica',2],'go')
show()




import matplotlib.pyplot as plt
import numpy as np


def f(t):
    return np.exp(-t) * np.cos(2 * np.pi * t)


t1 = np.arange(0, 5, 0.1)
t2 = np.arange(0, 5, 0.02)

plt.figure()
plt.subplot(441)
plt.plot(data[target=='setosa',0],data[target=='setosa',3],'bo')
plt.plot(data[target=='versicolor',0],data[target=='versicolor',3],'ro')
plt.plot(data[target=='versicolor',0],data[target=='virginica',3],'go')

plt.subplot(442)
plt.plot(data[target=='setosa',1],data[target=='setosa',3],'bo')
plt.plot(data[target=='versicolor',1],data[target=='versicolor',3],'ro')
plt.plot(data[target=='versicolor',1],data[target=='virginica',3],'go')

plt.subplot(443)
plt.plot(data[target=='setosa',2],data[target=='setosa',3],'bo')
plt.plot(data[target=='versicolor',2],data[target=='versicolor',3],'ro')
plt.plot(data[target=='versicolor',2],data[target=='virginica',3],'go')
plt.subplot(445)
plt.plot(data[target=='setosa',0],data[target=='setosa',2],'bo')
plt.plot(data[target=='versicolor',0],data[target=='versicolor',2],'ro')
plt.plot(data[target=='versicolor',0],data[target=='virginica',2],'go')
plt.subplot(446)
plt.plot(data[target=='setosa',1],data[target=='setosa',2],'bo')
plt.plot(data[target=='versicolor',1],data[target=='versicolor',2],'ro')
plt.plot(data[target=='versicolor',1],data[target=='virginica',2],'go')
plt.subplot(448)
plt.plot(data[target=='setosa',3],data[target=='setosa',2],'bo')
plt.plot(data[target=='versicolor',3],data[target=='versicolor',2],'ro')
plt.plot(data[target=='versicolor',3],data[target=='virginica',2],'go')
plt.subplot(449)
plt.plot(data[target=='setosa',0],data[target=='setosa',1],'bo')
plt.plot(data[target=='versicolor',0],data[target=='versicolor',1],'ro')
plt.plot(data[target=='versicolor',0],data[target=='virginica',1],'go')
plt.subplot(4,4,11)
plt.plot(data[target=='setosa',2],data[target=='setosa',1],'bo')
plt.plot(data[target=='versicolor',2],data[target=='versicolor',1],'ro')
plt.plot(data[target=='versicolor',2],data[target=='virginica',1],'go')
plt.subplot(4,4,12)
plt.plot(data[target=='setosa',3],data[target=='setosa',1],'bo')
plt.plot(data[target=='versicolor',3],data[target=='versicolor',1],'ro')
plt.plot(data[target=='versicolor',3],data[target=='virginica',1],'go')
plt.subplot(4,4,14)
plt.plot(data[target=='setosa',1],data[target=='setosa',0],'bo')
plt.plot(data[target=='versicolor',1],data[target=='versicolor',0],'ro')
plt.plot(data[target=='versicolor',1],data[target=='virginica',0],'go')
plt.subplot(4,4,15)
plt.plot(data[target=='setosa',2],data[target=='setosa',0],'bo')
plt.plot(data[target=='versicolor',2],data[target=='versicolor',0],'ro')
plt.plot(data[target=='versicolor',2],data[target=='virginica',0],'go')
plt.subplot(4,4,16)
plt.plot(data[target=='setosa',3],data[target=='setosa',0],'bo')
plt.plot(data[target=='versicolor',3],data[target=='versicolor',0],'ro')
plt.plot(data[target=='versicolor',3],data[target=='virginica',0],'go')


plt.show()


from pylab import figure, subplot, hist, xlim, show
xmin = min(data[:,0])
xmax = max(data[:,0])
figure()
subplot(411) # distribution of the setosa class (1st, on the top)
hist(data[target=='setosa',0],color='b',alpha=.7)
xlim(xmin,xmax)
subplot(412) # distribution of the versicolor class (2nd)
hist(data[target=='versicolor',0],color='r',alpha=.7)
xlim(xmin,xmax)
subplot(413) # distribution of the virginica class (3rd)
hist(data[target=='virginica',0],color='g',alpha=.7)
xlim(xmin,xmax)
subplot(414) # global histogram (4th, on the bottom)
hist(data[:,0],color='y',alpha=.7)
xlim(xmin,xmax)
show()


