import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd


def dataInfo(data):
    print('No. of features -> {}'.format(data.shape[1]))
    print('No. of numerical features -> {}, No. of categorical features -> {}'.format(data.select_dtypes(['float','int']).shape[1],
                                                                                      data.select_dtypes(['string']).shape[1]))
    getCategoricalColAndNunique(data)
    print('No. of Null Values -> {}'.format(data.isnull().sum().sum()))


def getCategoricalColAndNunique(data):
    for colName in data.columns:
        if (data[colName].dtype == np.float64) or (data[colName].dtype == np.int64):
            print('        Column Name -> {}, DataType -> {}'.format(colName, data[colName].dtype))
        else:
            print('        Column Name -> {}, DataType -> {}'.format(colName, data[colName].dtype))
            print('        Categorical NUnique -> {}'.format(data[colName].nunique()))

def plotKDEGraph(data):
    figure = plt.figure(figsize=(16, 8))
    for i in range(data.shape[1]-1):
        featureName = 'f_0{}'.format(i) if i<10 else 'f_{}'.format(i)
        plt.subplot(5, 6, i+1)
        if data[featureName].dtype == np.int64:
            sns.kdeplot(data[featureName], fill=True, color='red')
        else:
            sns.kdeplot(data[featureName], fill=True, color='blue')
        
    plt.tight_layout(h_pad=1, w_pad=0.5)
    plt.suptitle('Distribution Plots..')
    plt.show()
    
def grubbsTest(x, feature):
    n = len(x)
    mean_x = np.mean(x)
    sd_x = np.std(x)
    numerator = max(abs(x-mean_x))
    g_calculated = numerator/sd_x
    t_value = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
    g_critical = ((n - 1) * np.sqrt(np.square(t_value))) / (np.sqrt(n) * np.sqrt(n - 2 + np.square(t_value)))
    if g_critical > g_calculated: 
        result = 'Accepted'
        outlier_presence = False
    else:
        result = 'Rejected'  
        outlier_presence = True
        
    print('Feature: {}\t Hypothesis: {}'.format(feature, result))
    return outlier_presence
    
def Zscore_outlier(df):
    out=[]
    m = np.mean(df)
    sd = np.std(df)
    row = 0
    for i in df: 
        z = (i-m)/sd
        if np.abs(z) > 3: 
            out.append(row)
        row += 1
    return out
    
def detectOutliers(data):
    outlierFeatures = []
    for col in data.drop('id', axis=1).columns:
        if grubbsTest(data[col], col):
            outlierFeatures.append(col)
            
    for col in outlierFeatures: 
        percentage = round(100 * len(Zscore_outlier(df[col])) / df.shape[0], 2)
        print('Feature: {} \tPercentage of outliers: {}%'.format(col, percentage))
    

df = pd.read_csv('data.csv')
detectOutliers(df)