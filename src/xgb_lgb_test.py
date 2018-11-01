# coding=utf-8
"""
Xgboost and lightgbm test
"""

from preprocessing.npzLoader import loadData, loadNData
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier

if __name__ == '__main__':
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
    samples, labels = loadNData(data_dir, channels, 5)
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=0)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    xgb_param = {
        'clf__n_estimators': [20, 50, 100, 200],
        'clf__learning_rate': [0.01, 0.1, 0.99],
        'clf__objective': ['multi:softmax'],
        'clf__silent': [True]
    }

    lgb_param = {'clf__n_estimators': [20, 50, 100, 200],
                 'clf__learning_rate': [0.01, 0.1, 0.99],
                 'clf__objective': ['multiclass'],
                 'clf__silent': [True]
                 }

    ada_param = {'clf__n_estimators': [20, 50, 100, 200],
                 'clf__learning_rate': [0.01, 0.1, 0.99]
                 }


    params = [ada_param, xgb_param, lgb_param]
    classifiers = [AdaBoostClassifier(), XGBClassifier(), LGBMClassifier()]
    names = ['adaboost', 'xgboost', 'lightgbm']

    # grid search with cross validation
    for clf, param, name in zip(classifiers, params, names):
        print("Cross validate on classifier %s" % (name))
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



    # # cross validate
    # for clf, param, name in zip(classifiers, params, names):
    #     print("Cross validate on classifier %s" % (name))
    #     estimator = [('standardization', StandardScaler()), ('clf', clf)]
    #     pipe = Pipeline(estimator)
    #     cv_result = cross_validate(clf,
    #                                X_train, y_train,
    #                                cv=3,
    #                                n_jobs=-1,
    #                                verbose=3,
    #                                scoring='accuracy',
    #                                )
    #
    #     means = cv_result['test_score']
    #     fit_times = cv_result['fit_time']
    #     for mean, fitTime in zip(means, fit_times):
    #         print("%0.3f, fit time=%s, for %r" % (mean, fitTime, param))

