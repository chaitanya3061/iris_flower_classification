# from sklearn import datasets
# from sklearn.linear_model import LogisticRegression
# import pandas as pd
# iris = datasets.load_iris()
# features=pd.DataFrame(iris['data'])
# target=iris['target']
# def training_model():
#     model=LogisticRegression(max_iter=1000)
#     return model.fit(features,target)
import pandas as pd 
from sklearn.datasets import load_iris
data = load_iris()
# print(data.DESCR)

X = pd.DataFrame(data.data, columns=(data.feature_names))
y = pd.DataFrame(data.target, columns=['Target'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)

from sklearn.linear_model import LinearRegression

def training_model():
    model = LinearRegression()
    trained_model = model.fit(X_train, y_train)
    return trained_model