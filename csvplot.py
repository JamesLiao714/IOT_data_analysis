from matplotlib import pyplot as plt 
from matplotlib import style 
import numpy as np
style.use('ggplot')

x,y = np.read_csv('Accelerometer.csv', unpack = True, delimiter = ',')


plt.title('EPIC chart')
plt.ylabel('Y axis')
plt.xlable('x axis')

plot.show()