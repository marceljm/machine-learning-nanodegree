import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data/data.csv')
X = data[['x1','x2']]
y = data['y']

from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier()
classifier.fit(X,y)

plt.scatter(data['x1'], data['x2'], c=y, s=50)
plt.plot(y, classifier.predict(X))
plt.show()
