# coding=utf-8
"""
Test script for Machine learning methods
"""

from preprocessing.npzLoader import loadNData, loadData
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
import logging

if __name__ == '__main__':
    logging.basicConfig(filename='machine_learning_record.log', filemode='w', level=logging.DEBUG)

    feature_file = "../data/A6_exp_data/Features_00000020.npz"
    label_file = "../data/A6_exp_data/Labels_00000020.npz"
    data_dir = "../data/A6_exp_data"
    channels = [
        'EEG F3-A2',
        # 'EEG F4-A1',
        # 'EEG A1-A2',
        'EEG C3-A2',
        # 'EEG C4-A1',
        'EEG O1-A2',
        # 'EEG O2-A1',
        'EOG LOC-A2',
        # 'EOG ROC-A2'
    ]


    #samples, labels = loadData(feature_file, label_file, channels)
    samples, labels = loadNData(data_dir, channels, 5)
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=0)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    classifiers = [DecisionTreeClassifier(),
                  #DecisionTreeClassifier(class_weight='balanced'),
                  SVC(),
                  #SVC(class_weight='balanced'), #SVC with auto balance of class weight
                  KNeighborsClassifier(),
                  LogisticRegression(),
                  #LogisticRegression(class_weight='balanced'),
                  #GaussianProcessClassifier(),
                  MLPClassifier()
                  ]

    svc_param = {'clf__C': [0.1, 1.0, 10.0],
                 'clf__kernel': ['linear', 'rbf', 'poly'],
                 }
    dct_param = {'clf__criterion': ['gini', 'entropy'],
                 'clf__max_depth': [5, 20, 50],
                 }
    knb_param = {'clf__n_neighbors': [5, 10, 20]
                 }
    lr_param = {'clf__C': [0.1, 1.0, 10.0],
                #'clf__multi_class': ['multinomial']
                }
    gp_param = {'clf__multi_class': ['one_vs_rest', 'one_vs_one']
                }
    mlp_param = {'clf__hidden_layer_sizes': [(10), (30), (50), (15, 15), (25, 25), (10, 10, 10), (15, 15, 15)],
                 'clf__learning_rate': ['adaptive']
                 }


    params = [dct_param,
              #dct_param,
              svc_param,
              #svc_param,
              knb_param,
              lr_param,
              #lr_param,
              #gp_param,
              mlp_param
              ]

    names = ['decision tree',
             'support vector machine',
             'k_neighbors',
             'logistic regression',
             #'gaussian process',
             'multi-layer-perceptron'
             ]

    # grid search with cross validation
    for clf, param, name in zip(classifiers, params, names):
        print("Cross validate on classifier %s" % (name))
        logging.info("Cross validate on classifier %s" % (name))
        estimator = [('standardization', StandardScaler()), ('clf', clf)]
        pipe = Pipeline(estimator)
        clf_gs = GridSearchCV(pipe,
                              param_grid=param,
                              scoring='accuracy',
                              n_jobs=3,
                              cv=3,
                              verbose=3
                              )
        clf_gs.fit(X_train, y_train)
        means = clf_gs.cv_results_['mean_test_score']
        stds = clf_gs.cv_results_['std_test_score']
        fit_times = clf_gs.cv_results_['mean_fit_time']
        for mean, std, fitTime, params in zip(means, stds, fit_times, clf_gs.cv_results_['params']):
            print("%0.3f (+/-%0.03f), fit time=%s, for %r" % (mean, std * 2, fitTime, params))
            logging.info("%0.3f (+/-%0.03f), fit time=%s, for %r" % (mean, std * 2, fitTime, params))