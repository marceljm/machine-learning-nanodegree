import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data/2_class_data.csv')

plt.scatter(data['x1'], data['x2'], c=data['y'], s=50)
plt.show()
