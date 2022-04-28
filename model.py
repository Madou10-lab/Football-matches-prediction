import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class Model:

  def __init__(self):
    self.model1 = None
    self.model2 = None
    self.model3 = None

  def train(self,X,y):
    self.model1 = MLPClassifier(random_state=1, max_iter=300).fit(X, y)
    self.model2 = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    self.model3 = GaussianNB().fit(X, y)

  def predict(self,data,le):
    #data is list of data
    d = le.fit_transform(data[1:])
    data=np.array([np.append(int(data[0]),d)])

    #probabilities
    preds1 = self.model1.predict_proba(data.reshape(1, -1))
    preds2 = self.model2.predict_proba(data.reshape(1, -1))
    preds3 = self.model3.predict_proba(data.reshape(1, -1))

    return (preds1,preds2,preds3)
