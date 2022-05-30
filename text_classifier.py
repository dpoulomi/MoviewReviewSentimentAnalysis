#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import math
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn import preprocessing
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.utils import resample
nltk.download('punkt')
nltk.download('wordnet')
nltk.download("stopwords")
nltk.download('omw-1.4')

warnings.filterwarnings("ignore")


def clean_review(review):
    snowball = SnowballStemmer(language='english')
    no_html = BeautifulSoup(review).get_text()
    clean = re.sub("[^a-z\s]+", " ", no_html, flags=re.IGNORECASE)
    clean = re.sub("(\s+)", " ", clean)
    clean = clean.lower()
    stopwords_en = stopwords.words("english")
    cleaned_stopwords = [w for w in re.split("\W+", clean) if not w in stopwords_en]
    stemmed_words = []
    for w in cleaned_stopwords:
        stemmed_words.append(snowball.stem(w))
    return stemmed_words



# TASK 4 CELL
def naive_bayes_predict(review, logprior, loglikelihood):
    '''
    Params:
        review: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Return:
        total_prob: the sum of all the loglikelihoods of each word in the review (if found in the dictionary) + logprior (a number)

    '''
    
    
      # process the review to get a list of words
    word_l = clean_review(review)

    # initialize probability to zero
    total_prob = 0
    prediction = 0

    # add the logprior
    total_prob = total_prob + logprior   

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if (word in loglikelihood):
            # add the log likelihood of that word to the probability
            total_prob = total_prob + loglikelihood[word]
                   
    if(total_prob > 0.0):
        prediction = 1
    if(total_prob <= 0.0):
        prediction = 0

    return total_prob,prediction


# Defining main function
def main():
    clf_filename = 'naive_bayes_model_parameters.pkl'
    nb_clf = pickle.load(open(clf_filename, 'rb'))
    loglikelihood = nb_clf[1]
    my_review = "" 
    keep_going = True
    while keep_going:
        my_review = input ("Enter your review:")
        if my_review == "X":
            keep_going = False
            quit()
        if(keep_going):            
            output = ''
            probability , prediction = naive_bayes_predict(my_review, nb_clf[0], nb_clf[1]) 
            if(prediction == 1):
                output = 'negative'
            else:
                output = 'positive'
            print('The predicted output is', prediction , output )
            print('The probability for the input is', probability)
            input_tokens = my_review.split()
            for token in input_tokens:
                
                if token in loglikelihood:
                    print( token, loglikelihood[token], sep = '- ')
                else:
                    print(token, 0, sep = '- ')
  
# __name__
if __name__=="__main__":
    main()
    
    


# In[ ]:




