import numpy as np
import pandas as pd
import pyod
from pyod.models.vae import VAE
from pyod.models.iforest import IForest

def scan_for_new_anomalies(y_pred,n_old):
    """Function that scans for new anomalies
        based on how many anomalies where in the last
        scan.  Simply returns a boolean True if there
        is a new anomaly."""
    return 1 in y_pred[n_old:]

def fit_VAE_direct(df):
    """This is the function that performs unsupervised anomaly detection\
        on the scaled tweet data from a user."""
    #dataframe to array
    X = df.values
    ndim = X.shape[1] #the number of features
    random_state = np.random.RandomState(81)#Random seed
    outlier_fraction = 0.007 #.7% of all tweets are outliers (best fit)
    classifiers = {
        'Variational Auto Encoder (VAE)': VAE(epochs=20,
                contamination = outlier_fraction, random_state = random_state,
                encoder_neurons = [ndim,max(int(ndim/2),1),max(int(ndim/4),1)],
                decoder_neurons = [max(int(ndim/4),1),max(int(ndim/2),1),ndim],
                verbosity=0)
    }
    for i, (clf_name,clf) in enumerate(classifiers.items()):
        clf.fit(X)
        y_pred = clf.predict(X)
    return y_pred
    
def fit_VAE(username):
    """This is the function that performs unsupervised anomaly detection\
        on the scaled tweet data from a user."""
    #read the data in
    df = pd.read_csv('../data/processed/'+username+\
                '_scaled_tweet_features.csv').drop('Unnamed: 0',axis='columns')
    #dataframe to array
    X = df.values
    ndim = X.shape[1] #the number of features
    random_state = np.random.RandomState(81)#Random seed
    outlier_fraction = 0.007 #.7% of all tweets are outliers (best fit)
    classifiers = {
        'Variational Auto Encoder (VAE)': VAE(epochs=20,
                contamination = outlier_fraction, random_state = random_state,
                encoder_neurons = [ndim,max(int(ndim/2),1),max(int(ndim/4),1)],
                decoder_neurons = [max(int(ndim/4),1),max(int(ndim/2),1),ndim],
                verbosity=0)
    }
    for i, (clf_name,clf) in enumerate(classifiers.items()):
        clf.fit(X)
        y_pred = clf.predict(X)
    return y_pred
    
def fit_VAE_with_scores(username):
    """This is the function that performs unsupervised anomaly detection\
        on the scaled tweet data from a user."""
    #read the data in
    df = pd.read_csv('../data/processed/'+username+\
                '_scaled_tweet_features.csv').drop('Unnamed: 0',axis='columns')
    #dataframe to array
    X = df.values
    ndim = X.shape[1] #the number of features
    random_state = np.random.RandomState(81)#Random seed
    outlier_fraction = 0.01 #1% of all tweets are outliers
    classifiers = {
        'Variational Auto Encoder (VAE)': VAE(epochs=20,
                contamination = outlier_fraction, random_state = random_state,
                encoder_neurons = [ndim,max(int(ndim/2),1),max(int(ndim/4),1)],
                decoder_neurons = [max(int(ndim/4),1),max(int(ndim/2),1),ndim],
                verbosity=0)
    }
    for i, (clf_name,clf) in enumerate(classifiers.items()):
        clf.fit(X)
        scores_pred = clf.decision_function(X) * -1
        y_pred = clf.predict(X)
    return y_pred,scores_pred
    
def fit_multiple_with_scores(username):
    """This is the function that performs unsupervised anomaly detection\
        on the scaled tweet data from a user."""
    #read the data in
    df = pd.read_csv('../data/processed/'+username+\
                '_scaled_tweet_features.csv').drop('Unnamed: 0',axis='columns')
    #dataframe to array
    X = df.values
    ndim = X.shape[1] #the number of features
    random_state = np.random.RandomState(81)#Random seed
    outlier_fraction = 0.01 #1% of all tweets are outliers
    classifiers = {
        'Variational Auto Encoder (VAE)': VAE(epochs=20,
                contamination = outlier_fraction, random_state = random_state,
                encoder_neurons = [ndim,max(int(ndim/2),1),max(int(ndim/4),1)],
                decoder_neurons = [max(int(ndim/4),1),max(int(ndim/2),1),ndim],
                verbosity=0),
        'Isolation Forest': IForest(contamination=outlier_fraction,
                                random_state=random_state)
    }
    pred_out = np.array([])
    score_out = np.array([])
    for i, (clf_name,clf) in enumerate(classifiers.items()):
        clf.fit(X)
        scores_pred = clf.decision_function(X) * -1
        y_pred = clf.predict(X)
        if pred_out.shape == (0,):
            pred_out = y_pred.reshape(1,len(y_pred))
            score_out = scores_pred.reshape(1,len(score_out))
        else:
            pred_out = np.vstack((y_pred.reshape(1,len(y_pred)),pred_out))
            score_out = np.vstack((scores_pred.reshape(1,len(score_out)),
                                    score_out))
    return pred_out,score_out

if __name__ == '__main__':
    main()