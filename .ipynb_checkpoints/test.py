import evopipe

from sklearn.model_selection import train_test_split
from sklearn import datasets

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer
from sklearn.decomposition import PCA, KernelPCA, NMF
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


preproc_l = [MinMaxScaler(), StandardScaler(), Normalizer(), Binarizer(), PCA(), KernelPCA(), NMF(), SelectKBest(chi2)]
classif_l = [SVC(), LinearSVC(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(),
             AdaBoostClassifier(), QuadraticDiscriminantAnalysis()]
wine = datasets.load_wine()
train_X, test_X, train_Y, test_Y = train_test_split(wine.data, wine.target, test_size = 0.25, random_state = 0)

clf = evopipe.EvoPipeClassifier(preproc_l, classif_l, 2)
clf.fit(train_X, train_Y)

score = clf.score(test_X, test_Y)
print(score)

best_pipes = clf.best_pipelines()

for pipe, score in best_pipes:
    print("Score: {}, Pipe: {}".format(score, pipe.steps))