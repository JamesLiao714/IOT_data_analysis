 
import matplotlib.pyplot as plt 
#matplotlib.use('TKAgg')
import numpy as np
import pandas as pd
import csv



def plotData(plt, data, color):
	x = [p[0] for p in data]
	y = [p[1] for p in data]

	plt.xlabel('ms')
	plt.ylabel('val')
	plt.plot(x, y, 'go')
	plt.legend(loc = 'best',prop={'size':6})
	plt.savefig("output.png", dpi=300)

print('in')
x= []
y =[]
y1 = []
y2 = []
s = []
sy = []
w =[]
wy = []
Y = []
Yy = []
z = []
zy = []
flag = 0
num = 0
with open('Accelerometer.csv', newline='') as csvfile:
 	inF = csv.reader(csvfile)
 	for row in inF:
 		if num!=0 :
	 		x.append(float(row[0]))
	 		y.append(float(row[1]))
	 		y1.append(float(row[2]))
	 		y2.append(float(row[3]))
	 		if str(row[4]) == 'S':
	 			s.append(float(row[0]))
	 			sy.append(19)
	 		elif str(row[4]) == 'W':
	 			w.append(float(row[0]))
	 			wy.append(19)
	 		elif str(row[4]) == 'Y':
	 			Y.append(float(row[0]))
	 			Yy.append(19)
	 		
	 		msg = str(x[num-1]) + ' ' + str(y[num-1]) + ' ' + row[4]
	 		print(msg)
	 	num = num +1
 		

print(num)

fig=plt.figure(figsize=(6, 8),facecolor='white')

#plt.plot(x,y,'ro', label = 'x') #plot x
#plt.plot(x,y1,'bo',label = 'y') #plot y
plt.plot(x, y2,'go' ,label = 'z') #plot z
plt.plot(s, sy, label = 'study', c = 'yellow')
plt.plot(w, wy, label = 'walking', c = 'brown')
plt.plot(Y, Yy, label = 'cycling', c = 'cyan')
#plt.plot(z, zy, label = 'others', c = 'pink')
plt.ylabel('val')
plt.xlabel('ms')

my_x_ticks = np.arange(0, 8103751, 500000)
my_y_ticks = np.arange(-20, 20, 1)

#plt.xlim((0, 3545002))     #也可写成plt.xlim(-5, 5) 
#plt.ylim((-4, 4))     #也可写成plt.ylim(-2, 2)

plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.xticks(rotation= 45)
plt.title('Accelerometer\n')
plt.legend(loc = 'best',prop={'size':6})
plt.show()
fig.savefig("output.png")



'''colr = ["red", "green", "navy"]
  # 以迴圈輸出每一列
 data = []
 num = 0
 for row in inF:
 	print(row)
 	data.append(row)
 	num =num+1

 for n in range(3)
 	plotData(plt, result_data[n], color[n])
 plt.show()'''


'''trace1 = go.Scatter(
                    x=df['x'], y=df['logx'], # Data
                    mode='lines', name='logx' # Additional options
                   )
trace2 = go.Scatter(x=df['x'], y=df['sinx'], mode='lines', name='sinx' )
trace3 = go.Scatter(x=df['x'], y=df['cosx'], mode='lines', name='cosx')

layout = go.Layout(title='Simple Plot from csv data',
                   plot_bgcolor='rgb(230, 230,230)')

fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)

# Plot data in the notebook
py.iplot(fig, filename='simple-plot-from-csv')'''