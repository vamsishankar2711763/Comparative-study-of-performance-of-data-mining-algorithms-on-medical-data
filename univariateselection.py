import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# load data


dataframe = pandas.read_csv('data.csv')
list= ['Gender']

dataframe = dataframe.drop(list, axis=1)

array = dataframe.values
X = array[:,0:9]
Y = array[:,9]
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])

