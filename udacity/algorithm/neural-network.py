import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap

data = pd.read_csv('data/data.csv')
X = data[['x1','x2']]
y = data['y']

from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(max_iter=900)
classifier.fit(X,y)

# first we determine the grid of points -- i.e. the min and max for each of 
# the axises and then build a grid
resolution=0.02
x1_min, x1_max = X["x1"].min() - 0.2, X["x1"].max() + 0.2
x2_min, x2_max = X["x2"].min() - 0.2, X["x2"].max() + 0.2
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))

# setup marker generator and color map
markers = ('o', 'o')
colors = ('purple', 'yellow')
cmap = ListedColormap(colors[:len(np.unique(y))])

# plot the classifier decision boundaries
Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.1, cmap=cmap)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

# plot the data points
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X["x1"][y == cl].values, 
                y=X["x2"][y == cl].values,
                alpha=0.9, 
                c=cmap(idx),
                edgecolor='black',
                marker=markers[idx], 
                label=cl,
                s=50)    
plt.show()
