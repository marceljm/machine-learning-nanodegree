import matplotlib.pyplot as plt
import pandas as pd
import sys

columns = (i for i in range(1,16))
data = pd.read_csv('data/data.csv', names=columns, nrows=9)
print data

#show 15 figures
data.hist()

#show 1 figure (1-15)
#parameter = int(sys.argv[1])
#plt.hist(data[parameter])

plt.show()
