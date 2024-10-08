from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    #df = pd.read_excel('Sample2.xlsx', names = ['Lct','Reg','Hull','Oth','RMSE'])
    df = pd.read_excel('Sample2.xlsx', names = ['Lct','Reg','Hull','Rest','Oth','RMSE'])
    #df = pd.read_excel('Sample.xlsx', names = ['Lct','Reg','Rest','DOP','Out','Ord','RMSE'])
    X = df.iloc[:, :5]
    Y = df.iloc[:, 5]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, random_state=0)

    pipe_svc = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=0, criterion='gini'))

    param_grid = [{'randomforestclassifier__n_estimators': [50,100,150],
                    'randomforestclassifier__max_depth': [5,10,15]}
                    ]
    gs = GridSearchCV(estimator = pipe_svc,
                    param_grid = param_grid,
                    scoring = 'accuracy',
                    cv = 5,
                    refit = True,
                    n_jobs = -1)
    gs = gs.fit(X_train.values, Y_train.values)
    print(gs.best_score_)
    print(gs.best_params_)

    clf = gs.best_estimator_
    clf.fit(X_train.values,Y_train.values)

    joblib.dump(clf, 'model2.txt')

    print(accuracy_score(Y_test.values, clf.predict(X_test.values)))
    print(precision_score(Y_test.values, clf.predict(X_test.values), pos_label=1))
    print(recall_score(Y_test.values, clf.predict(X_test.values), pos_label=1))

    #feat_labels = np.array(['Lct','Reg','Rest','DOP','Out','Ord'])
    #pd.Series(clf.feature_importances_,index=feat_labels)

    #importances = clf.feature_importances_
    #indices = np.argsort(importances)[::-1]
    #plt.title('Feature Importances')
    #plt.bar(range(X_train.values.shape[1]),importances[indices],align='center')
    #plt.xticks(range(X_train.values.shape[1]),feat_labels[indices],rotation=90)
    #plt.xlim([-1,X_train.values.shape[1]])
    #plt.tight_layout()
    #plt.show()