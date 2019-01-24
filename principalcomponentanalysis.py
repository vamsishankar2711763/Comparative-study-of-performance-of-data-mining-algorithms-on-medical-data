import numpy
from pandas import read_csv
from sklearn.decomposition import PCA
# load data
dataframe = read_csv('data.csv')
list= ['Gender']

dataframe = dataframe.drop(list, axis=1)

array = dataframe.values
X = array[:,0:9]
Y = array[:,9]
# feature extraction
pca = PCA(n_components=5)
fit = pca.fit(X)
# summarize components
print (("Explained Variance: %s") % (fit.explained_variance_ratio_))
print (fit.components_)
