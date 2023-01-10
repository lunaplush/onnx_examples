import numpy as np
import pickle
import os


with open("logit_tfidf_btc_sentiment.pkl", "rb") as f:
     model =  pickle.load(f)