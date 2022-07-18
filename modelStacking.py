import pandas as pd
from kmeans import getNCluster, getSubmissionFile
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeanData = pd.read_csv('kmeansDataStack.csv')
gmmData = pd.read_csv('GMMDataStack.csv')
inputSubmissionFileName = 'sample_submission.csv'
outputDataFileName = 'mySubmissionFile.csv'

masterData = pd.concat([kmeanData, gmmData], axis=1, ignore_index=True)
kcluster = getNCluster(masterData)
print('Number of cluser for masterData -> {}'.format(kcluster))
model = KMeans(n_clusters=kcluster, init='k-means++')
model.fit(masterData)
labels = model.labels_
score = silhouette_score(masterData, labels, metric='euclidean', sample_size=100, random_state=200)
print("Silhoutte Score of Model Stack Model -> {}".format(score))
submissionFile = getSubmissionFile(inputSubmissionFileName, labels)
submissionFile.to_csv(outputDataFileName, index=False)
print('Submission File Created! ...')




