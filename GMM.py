from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import pandas as pd
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def getNCluster(data):
    model = KMeans()
    visualiser = KElbowVisualizer(model, k=(4,12))
    visualiser.fit(data)
    return visualiser.elbow_value_


def fitPipeline(pipelines, data):
    updatedPipeline = []
    for i, pipeline in enumerate(pipelines):
        print('Traning GMM for {} scaler...'.format(pipelineDict[i]))
        scaledData = pipeline['scaler'].fit_transform(data)
        modelObject = pipeline['model'].fit(scaledData)
        label = modelObject.predict(scaledData)
        pipeline['score'] = silhouette_score(scaledData, label, metric='euclidean', sample_size=100, random_state=200)
        pipeline['modelObj'] = modelObject
        updatedPipeline.append(pipeline)
    return updatedPipeline, label
    

def getSilhouetteScore(pipeline):
    for i, model in enumerate(pipeline):
        print('Model -> {}, Silhouette Score -> {}'.format(pipelineDict[i], model['score']))
        
def getBestModel(pipeline, pipeDict):
    bestModel = None
    bestModelName = ''
    bestScore = 0
    for i, model in enumerate(pipeline):
        if bestScore < model['score']:
            bestScore = model['score']
            bestModel = model['modelObj']
            bestModelName = pipeDict[i]
    return bestModelName, bestScore, bestModel

def getSubmissionFile(fileName, labels):
    df = pd.read_csv(fileName)
    df['Predicted'] = labels
    return df
    
def addPCAToBestModel(bestModelName, pipeDict, pipeline, data):
    bestModelDictIndex = list(pipeDict.keys())[list(pipeDict.values()).index(bestModelName)]
    pcaPipeline = pipeline[bestModelDictIndex]
    scaledData = pcaPipeline['scaler'].fit_transform(data)
    pcaData = PCA(n_components=10).fit_transform(scaledData)
    modelObject = pcaPipeline['model'].fit(pcaData)
    label = modelObject.predict(pcaData)
    pcaPipeline['score'] = silhouette_score(pcaData, label, metric='euclidean', sample_size=100, random_state=200)
    pcaPipeline['modelObj'] = modelObject
    return pcaPipeline, label
    
def selectedFeaturesModel(data, bestModelName, pipeDict, pipeline):
    bestModelDictIndex = list(pipeDict.keys())[list(pipeDict.values()).index(bestModelName)]
    reqPipeline = pipeline[bestModelDictIndex]
    firstHalf = ['f_0{}'.format(i) if i<10 else 'f_{}'.format(i) for i in range(7,14)]
    secondHalf = ['f_{}'.format(i) for i in range(22,29)]
    usefulFeaturesData = data[firstHalf+secondHalf]
    
    scaledData = reqPipeline['scaler'].fit_transform(usefulFeaturesData)
    modelObject = reqPipeline['model'].fit(scaledData)
    label = modelObject.predict(scaledData)
    reqPipeline['score'] = silhouette_score(scaledData, label, metric='euclidean', sample_size=100, random_state=200)
    reqPipeline['modelObj'] = modelObject
    return reqPipeline, label


def selectedFeaturesModelV2(data, bestModelName, pipeDict, pipeline):
    bestModelDictIndex = list(pipeDict.keys())[list(pipeDict.values()).index(bestModelName)]
    reqPipeline = pipeline[bestModelDictIndex]
    firstHalf = ['f_0{}'.format(i) if i<10 else 'f_{}'.format(i) for i in range(7,14)]
    secondHalf = ['f_{}'.format(i) for i in range(22,29)]
    
    scaledData1 = reqPipeline['scaler'].fit_transform(data[firstHalf])
    modelObject1 = reqPipeline['model'].fit(scaledData1)
    label1 = modelObject1.predict(scaledData1)
    label1 = pd.Series(label1)
    
    scaledData2 = reqPipeline['scaler'].fit_transform(data[secondHalf])
    modelObject2 = reqPipeline['model'].fit(scaledData2)
    label2 = modelObject2.predict(scaledData2)
    label2 = pd.Series(label2)
    finalData = pd.concat([label1, label2], axis=1, ignore_index=True)
    
    finalModelObject = reqPipeline['model'].fit(finalData)
    finalLabels = finalModelObject.predict(finalData)
    reqPipeline['score'] = silhouette_score(finalData, finalLabels, metric='euclidean', sample_size=100, random_state=200)
    reqPipeline['modelObj'] = finalModelObject
    return reqPipeline, finalLabels

def GMMDataStack(label1, label2, label3, label4):
    labelList = [label1, label2, label3, label4]
    df = pd.concat([pd.Series(x) for x in labelList], axis=1)
    df.to_csv('GMMDataStack.csv', index=False)
    
    
if __name__ == '__main__':
    
    inputDataFileName = 'data.csv'
    inputSubmissionFileName = 'sample_submission.csv'
    outputDataFileName = 'mySubmissionFile.csv'
    
    inputData = pd.read_csv(inputDataFileName)
    inputData.drop('id', axis=1, inplace=True)
    kCluster = getNCluster(inputData) 
    #clusterModel = GaussianMixture(n_components=kCluster, random_state=0)
    clusterModel = BayesianGaussianMixture(n_components=kCluster, random_state=42, max_iter=300, n_init=5)
    
    
    
    #pipelineMinMax = {'scaler': MinMaxScaler(), 'model': clusterModel, 'modelObj': None, 'score': None}
    #pipelineStandard = {'scaler': StandardScaler(), 'model': clusterModel, 'modelObj': None, 'score': None}
    pipelinePower = {'scaler': PowerTransformer(), 'model': clusterModel, 'modelObj': None, 'score': None}
    #pipelineQantile = {'scaler': QuantileTransformer(), 'model': clusterModel, 'modelObj': None, 'score': None}
    #pipelineRobust = {'scaler': RobustScaler(), 'model': clusterModel, 'modelObj': None, 'score': None}
    pipelineDict = {0:'MinMaxScaler', 1:'StandardScaler', 2:'PowerTransformer', 3:'QuantileTransformer', 4:'RobustScaler'}
    #finalPipelines = [pipelineMinMax, pipelineStandard, pipelinePower, pipelineQantile, pipelineRobust]
    finalPipelines = [pipelinePower]

    finalPipelines, labels = fitPipeline(finalPipelines, inputData)
    getSilhouetteScore(finalPipelines)
    # modelName, modelScore, model = getBestModel(finalPipelines, pipelineDict)
    # print('Best Model Name -> {} with score -> {}'.format(modelName, modelScore))
    # pcaPipe, pcalabels = addPCAToBestModel(modelName, pipelineDict, finalPipelines, inputData)
    # print('Best Model Score after adding PCA layer in best model -> {}'.format(pcaPipe['score']))
    # featureSelectionPipe, fslabels = selectedFeaturesModel(inputData, modelName, pipelineDict, finalPipelines)
    # print('Best Model Score after adding Feature Selection layer in best model -> {}'.format(featureSelectionPipe['score']))
    # featureSelectionPipeV2, labelsV2 = selectedFeaturesModelV2(inputData, modelName, pipelineDict, finalPipelines)
    # print('Best Model Score after adding Feature Selection layerV2 in best model -> {}'.format(featureSelectionPipeV2['score']))
    submissionFile = getSubmissionFile(inputSubmissionFileName, labels)
    submissionFile.to_csv(outputDataFileName, index=False)
    print('Submission File Created! ...')
    # GMMDataStack(labels, pcalabels, fslabels, labelsV2)
    # print('GMMDataStackReady!..')