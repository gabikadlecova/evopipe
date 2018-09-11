from sklearn import decomposition
from sklearn import feature_selection
from sklearn import preprocessing

from sklearn import svm
from sklearn import linear_model
from sklearn import discriminant_analysis
from sklearn import neural_network
from sklearn import naive_bayes
from sklearn import tree
from sklearn import neighbors
from sklearn import ensemble

preproc = {

    "PCA":          decomposition.PCA,
    "kBest":        feature_selection.SelectKBest,

}

clfs = {

    "SVC":          svm.SVC,
    "logR":         linear_model.LogisticRegression,
    "Perceptron":   linear_model.Perceptron,
    "SGD":          linear_model.SGDClassifier,
    "PAC":          linear_model.PassiveAggressiveClassifier,
    "LDA":          discriminant_analysis.LinearDiscriminantAnalysis,
    "QDA":          discriminant_analysis.QuadraticDiscriminantAnalysis,
    "MLP":          neural_network.MLPClassifier,
    "gaussianNB":   naive_bayes.GaussianNB,
    "DT":           tree.DecisionTreeClassifier,
}

feat_frac = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]

params = {
    'PCA': {
        # todo would be difficult now, ask probably
        # 'feat_frac': feat_frac,
        'whiten': [False, True],
    },


    'kBest': {
        # 'feat_frac': feat_frac,
    },
    'SVC': {
        'C': [0.1, 0.5, 1.0, 2, 5, 10, 15],
        'gamma': ['auto', 0.0001, 0.001, 0.01, 0.1, 0.5],
        'tol': [0.0001, 0.001, 0.01]
    },
    'logR': {
        'penalty': ['l2'],
        'C': [0.1, 0.5, 1.0, 2, 5, 10, 15],
        'tol': [0.0001, 0.001, 0.01],
        'solver': ['sag']
    },
    'Perceptron': {
        'penalty': ['None', 'l2', 'l1', 'elasticnet'],
        'n_iter': [1, 2, 5, 10, 100],
        'alpha': [0.0001, 0.001, 0.01]
    },
    'SGD': {
        'penalty': ['none', 'l2', 'l1', 'elasticnet'],
        'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
        'n_iter': [5, 10, 100],
        'alpha': [0.0001, 0.001, 0.01],
        'l1_ratio': [0, 0.15, 0.5, 1],
        'epsilon': [0.01, 0.05, 0.1, 0.5],
        'learning_rate': ['constant', 'optimal'],
        'eta0': [0.01, 0.1, 0.5], # wild guess
        'power_t': [0.1, 0.5, 1, 2] # dtto
    },
    'PAC': {
        'loss': ['hinge', 'squared_hinge'],
        'C': [0.1, 0.5, 1.0, 2, 5, 10, 15]
    },
    'LDA': {
        'solver': ['lsqr', 'eigen'],
        'shrinkage': [None, 'auto', 0.1, 0.5, 1.0]
    },
    'QDA': {
        'reg_param': [0.0, 0.1, 0.5, 1],
        'tol': [0.0001, 0.001, 0.01]
    },
    'MLP': {
        'activation': ['identity', 'logistic', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'tol': [0.0001, 0.001, 0.01],
        'max_iter': [10, 100, 200],
        'learning_rate_init': [0.0001, 0.001, 0.01],
        'power_t': [0.1, 0.5, 1, 2],
        'momentum': [0.1, 0.5, 0.9],
        'hidden_layer_sizes': [(100,), (50,), (20,), (10,)]
    },
    'DT': {
        'criterion': ['gini', 'entropy'],
        'max_features': [0.05, 0.1, 0.25, 0.5, 0.75, 1],
        'max_depth': [1, 2, 5, 10, 15, 25, 50, 100],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10, 20]
    },
    'gaussianNB': {},
    'copy': {},
    'kMeans': {},
    'union': {},
    'vote': {}
}