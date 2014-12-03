from pylab import *

m = 0.5
dp = 1.2
gaussian = lambda x: 1/(sqrt(2*pi)*dp)*exp(-(x-m)**2/(2*(dp**2)))
x = np.linspace(-5.0, 5.0, num=1000)
y = gaussian(x)
plot(x,y,'k',linewidth=3)
xlabel('x')
ylabel('y(x)')
axis([-5,5,0,0.8])
title('Gaussiana [media = '+str(m)+', desvio = '+str(dp)+']')
show()
