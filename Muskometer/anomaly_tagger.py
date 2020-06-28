import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyod.models.vae import VAE

def identify_anomalies(X,outlier_fraction = 0.007,epochs=20,path,username):
    """A function that performs variational auto encoding analysis on the tweet data"""
    ndim = X.shape[1] #the number of features
    random_state = np.random.RandomState(42)
    #outlier_fraction = 0.01 #1% of all tweets are outliers
    #specifies the model parameters
    classifiers = {
        'Variational Auto Encoder (VAE)': VAE(epochs,
                contamination = outlier_fraction, random_state = random_state,
                encoder_neurons = [ndim,max(int(ndim/2),1),max(int(ndim/4),1)],
                decoder_neurons = [max(int(ndim/4),1),max(int(ndim/2),1),20],
                verbosity=0)
    }

    for i, (clf_name,clf) in enumerate(classifiers.items()):
        clf.fit(X) #fits the model
        scores_pred = clf.decision_function(X) * -1 #model scores
        y_pred = clf.predict(X) #model predictions for anomalies
    return y_pred

# Don't forget to do this at some point:
#unscaled_tweet_features_df['anomalous'] = y_pred
#unscaled_tweet_features_df.to_csv(path+username+
#                                  '_anomaly_tagged_tweet_features.csv')