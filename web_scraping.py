# -*- coding: utf-8 -*-
"""
Created on Sun May 17 23:35:33 2020

@author: User
"""

#web scraping
import requests
from bs4 import BeautifulSoup
from time import time, sleep
from random import randint

#translation
from googletrans import Translator

#Utilities
from tqdm import tqdm
import numpy as np
import pandas as pd

#nlp
from nltk import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import mark_negation


#Download resources
import nltk

nltk.download("punkt")
nltk.download("vader_lexicon")

def honestdoc_comment(url):
    """
    This function is to scrap data from a webboard (https://www.honestdocs.com).

    INPUT
    url : String
      URL of the target website
    
    OUTPUT
    comment : List
      List of comments
    score : List
      List of rating score
    """
    #create connection
    data = requests.get(url)
    print("requests code : {}".format(data.status_code)) 
    print("note\n2xx: success\n4xx, 5xx: error")
    
    #scrape comment and score
    start_time = time() #start scraping data from page1
    r = requests.get(url, params=dict(query="web scraping",page=1)) 
    soup = BeautifulSoup(r.text,"html.parser")
    n = len(soup.find_all("div",{"class":"comments__content"})) #find n of items in the page
    
    #extract each item
    comment = [soup.find_all("div",
                             {"class":"comments__content"})[i].get_text().strip() for i in range(0,n)]
    score = [soup.find_all("span",
                           {"class":"stars star-rating"})[i].attrs["data-score"] for i in range(0,n)]
    elapsed_time = time() - start_time #finish scraping data from page1
    print("Time used for scraping data from page - 1 : {} s".format(elapsed_time))
    sleep(randint(1,3)) #mimic human behavior
           
    p = 2 #start scraping data from page2
    while n > 0: #until the number of items in a page = 0
        start_time = time() 
        r = requests.get(url, params=dict(query="web scraping",page=p))
        soup = BeautifulSoup(r.text,"html.parser")
        n = len(soup.find_all("div",{"class":"comments__content"}))
        [comment.append(soup.find_all("div",
                                      {"class":"comments__content"})[i].get_text().strip()) for i in range(0,n)]
        [score.append(soup.find_all("span",
                                    {"class":"stars star-rating"})[i].attrs["data-score"]) for i in range(0,n)]
        elapsed_time = time() - start_time
        print("Time used for scraping data from page - {} : {} s".format(p, elapsed_time))
        p +=1
        sleep(randint(1,3))
    
    #backup data 
    pd.DataFrame({"comment": comment, 
                  "score": score}).to_csv("comment_"+str(url[url.rfind("/")+1:]) + ".csv", index=False)
    
    return comment, score


comments, scores = honestdoc_comment(r"https://www.honestdocs.co/hospitals/ramathibodi-hospital")

# restore data
rama= pd.read_csv("comment_ramathibodi-hospital.csv")
rama.to_csv("comment_ramathibodi-hospital.csv", index=False)

## Translate from Thais to English
def th2en(comment):
  return Translator().translate(comment, src="th", dest="en").text

tqdm.pandas()

rama["en"] = rama.progress_apply(lambda x: th2en(x["comment"]), axis=1)

rama.to_csv("en_comment_ramathibodi-hospital.csv", index=False)
##END
