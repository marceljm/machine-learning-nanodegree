import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

columns = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
data = pd.read_csv('data/data.csv', names=columns)

plt.scatter(data[1], np.arange(1, data[1].size + 1, 1))
plt.show()
