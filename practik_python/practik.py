import numpy as np          
import matplotlib.pyplot as plt   
from mpl_toolkits.mplot3d import axes3d
import pylab
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy

# Пример построения графика функции:

# x = np.arange(-10, 10.01, 0.01)
# plt.plot(x, x**2)
# plt.show()


# x = np.arange(-10, 10.01, 0.01)
# plt.plot(x, np.sin(x), x, np.cos(x), x, -x)
# plt.show()


# x = np.arange(-10, 10.01, 0.01)
# plt.plot(x, np.sin(x), x, np.cos(x), x, -x)
# plt.xlabel(r'$x$')
# plt.ylabel(r'$f(x)$')
# plt.title(r'$f_1(x)=\sin(x),\ f_2(x)=\cos(x),\ f_3(x)=-x$')
# plt.grid(True)
# plt.show()


# x = np.arange(-10, 10.01, 0.01)
# plt.figure(figsize=(10, 5))
# plt.plot(x, np.sin(x), label=r'$f_1(x)=\sin(x)$')
# plt.plot(x, np.cos(x), label=r'$f_2(x)=\cos(x)$')
# plt.plot(x, -x, label=r'$f_3(x)=-x$')
# plt.xlabel(r'$x$', fontsize=14)
# plt.ylabel(r'$f(x)$', fontsize=14)
# plt.grid(True)
# plt.legend(loc='best', fontsize=12)
# plt.savefig('figure_with_legend.png')
# plt.show()


# График может быть построен в полярной системе координат, для этого при создании subplot необходимо указать параметр polar=True:


# plt.subplot(111, polar=True)
# phi = np.arange(0, 2*np.pi, 0.01)
# rho = 2*phi
# plt.plot(phi, rho, lw=2)
# plt.show()


# Или может быть задан в параметрической форме (для этого не требуется никаких дополнительных действий, поскольку два массива, которые передаются в функцию plot воспринимаются просто как списки координат точек, из которых состоит график):


# t = np.arange(0, 2*np.pi, 0.01)
# r = 4
# plt.plot(r*np.sin(t), r*np.cos(t), lw=3)
# plt.axis('equal')
# plt.show()


# ax = axes3d.Axes3D(plt.figure())
# i = np.arange(-1, 1, 0.01)
# X, Y = np.meshgrid(i, i)
# Z = X**2 - Y**2
# ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
# plt.show()



# mu, sigma = 100, 15
# x = mu + sigma * np.random.randn(10000)
# # the histogram of the data
# n, bins, patches = plt.hist(x, 50, density=True, facecolor='g', alpha=0.75)

# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# plt.title('Histogram of IQ')
# plt.text(60, .030, r'$\mu=100,\ \sigma=15$')
# plt.text(50, .033, r'$\varphi_{\mu,\sigma^2}(x) = \frac{1}{\sigma\sqrt{2\pi}} \,e^{ -\frac{(x- \mu)^2}{2\sigma^2}} = \frac{1}{\sigma} \varphi\left(\frac{x - \mu}{\sigma}\right),\quad x\in\mathbb{R}$', fontsize=20, color='red')
# plt.axis([40, 160, 0, 0.04])
# plt.grid(True)
# plt.show()


# Каждую последовательность можно отобразить своим типом точек:

# равномерно распределённые значения от 0 до 5, с шагом 0.2
# t = np.arange(0., 5., 0.2)

# # красные чёрточки, синие квадраты и зелёные треугольники
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# plt.show()


# Также в matplotlib существует возможность строить круговые диаграммы:

# data = [33, 25, 20, 12, 10]
# plt.figure(num=1, figsize=(6, 6))
# plt.axes(aspect=1)
# plt.title('Plot 3', size=14)
# plt.pie(data, labels=('Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5'))
# plt.show()


# И аналогичным образом столбчатые диаграммы:

# objects = ('A', 'B', 'C', 'D', 'E', 'F')
# y_pos = np.arange(len(objects))
# performance = [10,8,6,4,2,1]

# plt.bar(y_pos, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, objects)
# plt.ylabel('Value')
# plt.title('Bar title')

# plt.show()


def makeData():
        x = numpy.arange(-10, 10, 0.1)
        y = numpy.arange(-10, 10, 0.1)
        xgrid, ygrid = numpy.meshgrid(x, y)
        zgrid = numpy.sin(xgrid)*numpy.sin(ygrid)/(xgrid*ygrid)
        return xgrid, ygrid, zgrid

x, y, z = makeData()

fig = pylab.figure()
axes = Axes3D(fig)
axes.plot_surface(x, y, z, rstride=4, cstride=4, cmap=cm.jet)
pylab.show()