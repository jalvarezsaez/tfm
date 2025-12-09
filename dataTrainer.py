import numpy as np
import seaborn as sns
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline


def evaluate_heatmap(ytest, pred):
    sns.heatmap(confusion_matrix(ytest, pred), annot=True, cmap='coolwarm')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Confusion Matrix')
    plt.show()


def evaluate_auc(ytest, pred):
    fpr, tpr, thresholds = roc_curve(ytest, pred)
    auc = roc_auc_score(ytest, pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay(fpr=fpr, tpr=tpr, name=f"LogReg (AUC={auc:.3f})").plot(ax=ax)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='(AUC=0.5)')
    ax.set_title('Curva ROC')
    ax.legend(loc='lower right')
    plt.show()


def print_evaluation_results(ytest, pred):
    print('Accuracy Score: ', accuracy_score(ytest, pred))
    print('Mean Absolute Error: ', mean_absolute_error(ytest, pred))
    print('Mean Squared Error: ', mean_squared_error(ytest, pred))
    print('R2 Score: ', r2_score(ytest, pred))


def train_predict_Logistic_Regression(xtrain, ytrain, xtest, ytest):
    lr = LogisticRegression()
    lr.fit(xtrain, ytrain)
    print(lr.score(xtrain, ytrain))
    lr_prediction = lr.predict(xtest)

    evaluate_heatmap(ytest, lr_prediction)
    evaluate_auc(ytest, lr_prediction)

    print_evaluation_results(ytest, lr_prediction)

def train_predict_svm(xtrain, ytrain, xtest, ytest):
    svm = SVC(kernel='linear', random_state=0)
    svm.fit(xtrain, ytrain)
    svm_prediction = svm.predict(xtest)
    print(svm.score(xtest, ytest))

    evaluate_heatmap(ytest, svm_prediction)
    evaluate_auc(ytest, svm_prediction)

    print_evaluation_results(ytest, svm_prediction)


def train_predict_lgbmc(xtrain, ytrain, xtest, ytest):
    clf = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42)
    clf.fit(xtrain, ytrain)
    lgbmc_prediction = clf.predict(xtest)
    print(clf.score(xtest, ytest))

    evaluate_heatmap(ytest, lgbmc_prediction)
    evaluate_auc(ytest, lgbmc_prediction)

    print_evaluation_results(ytest, lgbmc_prediction)


def optimise_svm(xtrain, ytrain, xtest, ytest):

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='linear', class_weight='balanced'))
    ])

    param_grid = {
        'svc__C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
        'svc__gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1]
    }

    # Validación estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='roc_auc',  # ajusta según tu caso: 'roc_auc', 'balanced_accuracy', etc.
        n_jobs=-1,
        verbose=1
    )

    grid.fit(xtrain, ytrain)
    print("Mejores parámetros:", grid.best_params_)
    print("Mejor AUC:", grid.best_score_)

    best_prediction = grid.predict(xtest)
    evaluate_heatmap(ytest, best_prediction)
    evaluate_auc(ytest, best_prediction)

    print_evaluation_results(ytest, best_prediction)
