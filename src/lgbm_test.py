# coding=utf-8
"""
test script for lightgbm method
"""

from preprocessing.npzLoader import loadData, loadNData
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import logging

if __name__ == '__main__':
    logging.basicConfig(filename='lgbm_record.log', filemode='w', level=logging.DEBUG)

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
    # X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=0)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    lgb_param = {'clf__n_estimators': [25, 50, 75, 100],
                 'clf__learning_rate': [0.75, 1, 1.25, 1.5],
                 'clf__objective': ['multiclass'],
                 'clf__num_leaves': [10, 20, 30],
                 'clf__max_depth': [5, 10, 20],
                 'clf__class_weight': ['balanced', None],
                 #'clf__min_child_samples': [10],
                 'clf__silent': [False]
                 }

    estimator = [('standardization', StandardScaler()), ('clf', LGBMClassifier())]
    pipe = Pipeline(estimator)
    clf_gs = GridSearchCV(pipe,
                          param_grid=lgb_param,
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