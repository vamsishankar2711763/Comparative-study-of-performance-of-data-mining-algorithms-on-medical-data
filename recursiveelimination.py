from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load data



dataframe = read_csv('data.csv')
list= ['Gender']

dataframe = dataframe.drop(list, axis=1)
print(dataframe)
array = dataframe.values
X = array[:,0:9]
Y = array[:,9]
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 7)
fit = rfe.fit(X, Y)
print (("Num Features: %d") % (fit.n_features_))
print (("Selected Features: %s") % (fit.support_))
print (("Feature Ranking: %s") % (fit.ranking_))
