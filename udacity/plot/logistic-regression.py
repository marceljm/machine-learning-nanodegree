import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data/2_class_data.csv')
X = data[['x1','x2']]
y = data['y']

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X,y)

plt.scatter(data['x1'], data['x2'], c=y, s=50)
plt.plot(y, classifier.predict(X))
plt.show()
