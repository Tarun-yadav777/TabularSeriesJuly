import numpy as np 
import pandas as pd 
import os
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, roc_auc_score, balanced_accuracy_score
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoost, Pool
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import gc
from GMM import getSubmissionFile

df =  pd.read_csv('data.csv')

def getTransformedData(data):
    data.drop('id', axis=1, inplace=True)
    data = data[usefulFeatures]
    scaledData = PowerTransformer().fit_transform(data)
    scaledData = pd.DataFrame(scaledData, columns=usefulFeatures)
    return scaledData

scores = []
usefulFeatures= ['f_07', 'f_08', 'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_27', 'f_28']
scaledData = getTransformedData(df)

def getLabelScores(labels, data, model):
    silScore = silhouette_score(data, labels)
    calScore = calinski_harabasz_score(data, labels)
    davScore = davies_bouldin_score(data, labels)
    
    scoreList = [model, silScore, calScore, davScore]
    
    print('Silhouette : {} | Calinski Harabasz : {} | Davies Bouldin : {}'.format(silScore, calScore, davScore))
    return scoreList

def getScore(labels, preds, probas):
    score = (balanced_accuracy_score(labels, preds),roc_auc_score(labels, probas, average="weighted", multi_class="ovo"))
    return score

print('Training BGMM moel...')
bgmModel = BayesianGaussianMixture(n_components=7, random_state=0, n_init=10).fit(scaledData)
bgmPredictProbabs = bgmModel.predict_proba(scaledData)
bgmPredict = np.argmax(bgmPredictProbabs, axis=1)
scores.append(getLabelScores(bgmPredict, scaledData, 'BGM Model'))

scaledData['predict'] = bgmPredict
scaledData['predict_proba'] = 0
for n in range(7):
    scaledData[f'predict_proba_{n}'] = bgmPredictProbabs[:, n]
    scaledData.loc[scaledData['predict']==n, 'predict_proba'] = scaledData[f'predict_proba_{n}']
    
idxs = np.array([])
for n in range(7):
    median = scaledData[scaledData.predict==n]['predict_proba'].median()
    idx = scaledData[(scaledData.predict==n) & (scaledData.predict_proba > 0.70)].index
    idxs = np.concatenate((idxs, idx))
    print(f'Class n{n}  |  Median : {median:.4f}  |  Training data : {len(idx)/len(scaledData[(scaledData.predict==n)]):.1%}')

X = scaledData.loc[idxs][usefulFeatures].reset_index(drop=True)
y = scaledData.loc[idxs]['predict'].reset_index(drop=True)

params_lgb = {
    'objective': 'multiclass',
    'boosting': 'gbdt',
    'learning_rate': 4e-2,
    'verbosity': -1,
    'n_jobs': -1,
    'num_classes': 7,
    'random_state': 0
}

params_xgb = {
    'booster': 'gbtree',
    'objective': 'multi:softprob',
    'learning_rate': 4e-2,
    'num_class': 7,
    'seed': 0,
    'gpu_id': 0,
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor'
    }

params_ctb = {
    'objective': 'MultiClass',
    'bootstrap_type': 'Poisson',
    #'boosting_type': 'Ordered',  # or 'Plain'
    'classes_count': 7,
    'num_boost_round': 20000,
    'learning_rate': 4e-1,
    'random_seed': 0,
    'task_type': 'GPU'
    
}

lgb_predict_proba = 0
xgb_predict_proba = 0
ctb_predict_proba = 0
etc_predict_proba = 0
qda_predict_proba = 0
gnb_predict_proba = 0
svc_predict_proba = 0
knc_predict_proba = 0
classif_scores = []

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0,)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"===== fold{fold} =====")
    X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
    X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

    # LightGBM
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid)

    ES = lgb.early_stopping(stopping_rounds=300, verbose=False)
    model = lgb.train(params=params_lgb, 
                      train_set=lgb_train,
                      valid_sets=lgb_valid, 
                      num_boost_round = 5000, 
                      callbacks = [ES])

    y_pred_proba = model.predict(X_valid)
    y_pred = np.argmax(y_pred_proba, axis=1)

    s = getScore(y_valid, y_pred, y_pred_proba)
    print(f"LightGBM   AUC : {s[1]:.3f} | Accuracy : {s[0]:.1%}")
    classif_scores.append(s)

    lgb_predict_proba += model.predict(scaledData[usefulFeatures]) / 10

    del lgb_train, lgb_valid, model, s, y_pred, y_pred_proba
    gc.collect()

    # XGBoost
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_valid = xgb.DMatrix(X_valid, label=y_valid)

    model = xgb.train(params_xgb,
                      dtrain=xgb_train,
                      evals=[(xgb_train, 'train'),(xgb_valid, 'eval')],
                      verbose_eval=False,
                      num_boost_round=5000,
                      early_stopping_rounds=300,
                     )

    y_pred_proba = model.predict(xgb_valid)
    y_pred = np.argmax(y_pred_proba, axis=1)

    s = getScore(y_valid, y_pred, y_pred_proba)
    print(f"XGBoost    AUC : {s[1]:.3f} | Accuracy : {s[0]:.1%}")
    classif_scores.append(s)

    xgb_predict_proba += model.predict(
        xgb.DMatrix(scaledData[usefulFeatures]),
    ) / 10

    del xgb_train, xgb_valid, model, s, y_pred, y_pred_proba
    gc.collect()
    
    # ExtraTreesClassifier
    model = ExtraTreesClassifier(n_estimators=300, random_state=0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    y_pred_proba = model.predict_proba(X_valid)

    s = getScore(y_valid, y_pred, y_pred_proba)
    print(f"ExtraTree  AUC : {s[1]:.3f} | Accuracy : {s[0]:.1%}")
    classif_scores.append(s)

    etc_predict_proba += model.predict_proba(scaledData[usefulFeatures]) / 10

    del model, s, y_pred, y_pred_proba
    gc.collect()

    # SVC
    model = SVC(probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    y_pred_proba = model.predict_proba(X_valid)

    s = getScore(y_valid, y_pred, y_pred_proba)
    print(f"SVC        AUC : {s[1]:.3f} | Accuracy : {s[0]:.1%}")
    classif_scores.append(s)

    svc_predict_proba += model.predict_proba(scaledData[usefulFeatures]) / 10

    del model, s, y_pred, y_pred_proba
    gc.collect()

    # KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=20)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    y_pred_proba = model.predict_proba(X_valid)

    s = getScore(y_valid, y_pred, y_pred_proba)
    print(f"KNeighbors AUC : {s[1]:.3f} | Accuracy : {s[0]:.1%}")
    classif_scores.append(s)

    knc_predict_proba += model.predict_proba(scaledData[usefulFeatures]) / 10

    del model, s, y_pred, y_pred_proba
    gc.collect()

scores.append(getLabelScores(np.argmax(lgb_predict_proba, axis=1), scaledData, "LightGBM"))
scores.append(getLabelScores(np.argmax(xgb_predict_proba, axis=1), scaledData, "XGBoost"))
scores.append(getLabelScores(np.argmax(etc_predict_proba, axis=1), scaledData, "ExtraTrees"))
scores.append(getLabelScores(np.argmax(svc_predict_proba, axis=1), scaledData, "SVC"))
scores.append(getLabelScores(np.argmax(knc_predict_proba, axis=1), scaledData, "KNeighbors"))

performanceScoreDF = pd.DataFrame(classif_scores, columns = ["balanced_accuracy_score", "roc_auc_score"]).mean(0)
performanceScoreDF.to_csv('performanceScoreDF.csv')

def soft_voting(preds_probas):
    pred_test = np.zeros((df.shape[0], 7))
    
    for i, p in enumerate(preds_probas):
        pred_test += p[0] * p[1]
    
    return np.argmax(pred_test, axis=1)

sv_predict = soft_voting((
    (lgb_predict_proba, 0.0),
    (xgb_predict_proba, 0.75),
    (etc_predict_proba, 1.0),
    (svc_predict_proba, 1.0),
    (knc_predict_proba, 1.0),
))

scores.append(getLabelScores(sv_predict, scaledData, "Soft voting"))

allScores_df = pd.DataFrame(scores, columns=["Model", "silhouette", "Calinski_Harabasz", "Davis_Bouldin"])
allScores_df.to_csv("modelScores.csv", index=False)



submissionFile = getSubmissionFile('sample_submission.csv', sv_predict)
submissionFile.to_csv('mySubmissionFile.csv', index=False)
