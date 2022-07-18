import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from GMM import getSubmissionFile

def getTransformedData(data):
    data.drop('id', axis=1, inplace=True)
    data = data[usefulFeatures]
    scaledData = PowerTransformer().fit_transform(data)
    return scaledData

df = pd.read_csv('data.csv')
usefulFeatures= ['f_07', 'f_08', 'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_27', 'f_28']
scaledData = getTransformedData(df)
print('DataTransformed...')

bgmModel = BayesianGaussianMixture(n_components=7, random_state=0, n_init=10).fit(scaledData)
predictions = bgmModel.predict(scaledData)
df['predictedLabels'] = predictions
print('BGMM trained...')

pp = bgmModel.predict_proba(scaledData)
dfNew = pd.DataFrame(scaledData, columns = usefulFeatures)
dfNew[['predictionProbability_{}'.format(i) for i in range(7)]] = pp
dfNew['predictedLabels'] = predictions
dfNew['predictProbab'] = np.max(pp, axis=1)
dfNew['predicts'] = np.argmax(pp, axis=1)

trainIndex=np.array([])
for i in range(7):
    indexArr = dfNew[(dfNew['predictedLabels'] == i) & (dfNew['predictProbab'] > 0.80)].index
    trainIndex = np.concatenate((trainIndex, indexArr))
    
XNew = dfNew.loc[trainIndex][usefulFeatures] 
y = dfNew.loc[trainIndex]['predicts']

params_lgb = {'learning_rate': 0.07,'objective': 'multiclass','boosting': 'gbdt','n_jobs': -1,'verbosity': -1, 'num_classes':7}
lgbm_predict_proba = 0  
classif_scores = []

stratifyModel = StratifiedKFold(n_splits=10, shuffle=True, random_state = 0)
for fold, (trainIdx, validIdx) in enumerate(stratifyModel.split(XNew, y)):
    trainDataset = lgb.Dataset(XNew.iloc[trainIdx], y.iloc[trainIdx], feature_name=usefulFeatures)
    validDataset = lgb.Dataset(XNew.iloc[validIdx], y.iloc[validIdx], feature_name=usefulFeatures)
    
    model = lgb.train(params = params_lgb, 
                train_set = trainDataset, 
                valid_sets =  validDataset, 
                num_boost_round = 5000, 
                callbacks=[ lgb.early_stopping(stopping_rounds=300, verbose=False)])  
    
    y_pred_proba = model.predict(XNew.iloc[validIdx])
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    s = (balanced_accuracy_score(y.iloc[validIdx], y_pred),
        roc_auc_score(y.iloc[validIdx], y_pred_proba, average="weighted", multi_class="ovo"))
    classif_scores.append(s)

    lgbm_predict_proba += model.predict(dfNew[usefulFeatures]) / 10
    
print(pd.DataFrame(classif_scores, columns = ["balanced_accuracy_score", "roc_auc_score"]).mean(0))
 

    
finalLabels=np.argmax(lgbm_predict_proba,axis=1)
submissionFile = getSubmissionFile('sample_submission.csv', finalLabels)
submissionFile.to_csv('mySubmissionFile.csv', index=False)

