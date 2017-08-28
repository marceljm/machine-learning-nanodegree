import matplotlib.pyplot as plt
import pandas as pd

columns = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
data = pd.read_csv('data.csv', names=columns)
data.hist()

plt.show()
