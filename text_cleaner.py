#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import re, string, unicodedata
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk import word_tokenize, sent_tokenize, FreqDist
from nltk.corpus import stopwords, wordnet
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import preprocessor as p

class TextCleaner:
    '''
    
    '''
    
    def __init__(self):
        pass
        
    
    '''
    
    Parameters
    ----------
        
    list_of_text :
    
    Returns
    -------
    
    list
    '''    
    def clean(self, list_of_text):
        cleaned_text = []
        for x in list_of_text:
            text = self.lower_case(x)
            text = self.general_clean(text)
            text = self.remove_digits(text)
            text = self.remove_punctuation(text)
            text = self.tokenize_and_lemmatize(text)            
            cleaned_text.append(text)
            
        cleaned_text = self.remove_stop_words(cleaned_text)
        
        return cleaned_text
    
    '''
    Used to lower case the passed text.
    
    Parameters
    ----------
        
    text :
    
    Returns
    -------
    
    str
        text lower cased
    '''    
    def lower_case(self, text):
        return str(text.lower())
    
    ''' 
    From the tweet-preprocessor package using preprocessor to remove 
    URLs, Hashtags, Mentions, Reserved words (RT, FAV), Emojis, Smileys.
    
    Parameters
    ----------
        
    text :
    
    Returns
    -------
    
    str
        text with URLs, Hashtags, Mentions, Reserved words (RT, FAV), 
        Emojis, Smileys removed.
    '''
    def general_clean(self, text):
        return p.clean(text)
    
    '''
    Used to remove any numbers from the text.
    
    Parameters
    ----------
        
    text :
    
    Returns
    -------
    
    str
        text with digits removed.
    '''
    def remove_digits(self, text):
        return re.sub(r'\d+', '', text)
    
    '''
    Used to remove any punctuation from the text.
    
    Parameters
    ----------
        
    text :
    
    Returns
    -------
    
    str
        text with punctuations removed.
    '''
    def remove_punctuation(self, text):
        return re.sub(r'[^\w\s]', '', text)
    
    '''
    Used to tokenize and lemmatize the passed text.
    
    Parameters
    ----------
        
    text :
    
    Returns
    -------
    
    list
        a list of tokenized words extracted from the text
    '''
    def tokenize_and_lemmatize(self, text):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        tokenizer = TweetTokenizer()
        return [(lemmatizer.lemmatize(w)) for w in tokenizer.tokenize((text))]
    
    '''
    Used to remove stop words.
    
    Parameters
    ----------
        
    text : list
        tokenized version of the text
        
    Returns
    -------
    
    pandas.Dataframe
        a list of tokenized words with stop words removed
    '''
    def remove_stop_words(self, text):
        stop_words = set(stopwords.words('english'))
        text = pd.Series(text)
        text = text.apply(lambda x: [item for item in x if item not in stop_words])
        return text


# In[ ]:




