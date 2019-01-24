from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# load data

dataframe = read_csv('data.csv')
list= ['Gender']

dataframe = dataframe.drop(list, axis=1)

array = dataframe.values
X = array[:,0:9]
Y = array[:,9]
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)
