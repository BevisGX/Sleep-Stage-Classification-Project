# coding=utf-8
"""
multi-layer perceptron classifier test script
"""

from preprocessing.npzLoader import loadNData, loadData
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import logging

if __name__ == '__main__':
    logging.basicConfig(filename='multi_layer_perceptron_record.log', filemode='w', level=logging.DEBUG)

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


    samples, labels = loadData(feature_file, label_file, channels)
    #samples, labels = loadNData(data_dir, channels, 5)
    #X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=0)
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    size = []
    for i in range(10, 101, 10):
        for j in range(10, 101, 10):
            for k in range(10, 101, 10):
                size.append((i, j, k))
    mlp_param = {'clf__hidden_layer_sizes': size,
                 'clf__learning_rate': ['adaptive']
                 }

    estimator = [('standardization', StandardScaler()), ('clf', MLPClassifier())]
    pipe = Pipeline(estimator)
    clf_gs = GridSearchCV(pipe,
                          param_grid=mlp_param,
                          scoring='accuracy',
                          n_jobs=3,
                          cv=3,
                          verbose=3
                          )
    clf_gs.fit(samples, labels)
    means = clf_gs.cv_results_['mean_test_score']
    stds = clf_gs.cv_results_['std_test_score']
    fit_times = clf_gs.cv_results_['mean_fit_time']
    for mean, std, fitTime, params in zip(means, stds, fit_times, clf_gs.cv_results_['params']):
        print("%0.3f (+/-%0.03f), fit time=%s, for %r" % (mean, std * 2, fitTime, params))
        logging.info("%0.3f (+/-%0.03f), fit time=%s, for %r" % (mean, std * 2, fitTime, params))