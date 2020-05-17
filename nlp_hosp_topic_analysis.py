# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:47:07 2020

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#NLP liberies
import spacy
import collections
import re
import sys
import time
import itertools
from wordcloud import WordCloud, STOPWORDS 
from math import pi

import gensim
import nltk
from gensim.utils import simple_preprocess
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora.dictionary import Dictionary
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from pprint import pprint


### create dataframe from csv file loaded from web scrapping (honestdoc.com)
full_data = pd.read_csv("D:/_RADS611 NLP/Assignment/honest_doc_full_2.csv")
full_data.head()

### Overall score data visualization

df_v = full_data.copy()
del df_v['comment']
del df_v['en']
df_v.head()

mean = df_v.groupby('hospital').mean()
count = df_v.groupby('hospital').count()
Visual_df= pd.merge(mean, count, on='hospital')
Visual_df.columns = ['mean', 'count']
Visual_df.head()

count_30 = Visual_df[Visual_df['count']>=30]
count_30.head()

count_100 = Visual_df[Visual_df['count']>=100]
count_100.head()

plot = Visual_df.plot.scatter(x='count', y='mean')
plt.title('All hospitals')
plt.show()

plot = count_30.plot.scatter(x='count', y='mean')
plt.title('Hospitals that have at least 30 comments')
plt.show()

plot = count_100.plot.scatter(x='count', y='mean')
plt.title('Hospitals that have at least 100 comments')
plt.show()

# From the visualization, I decided to choose these hospitals
#siriraj =  "โรงพยาบาลศิริราช"
#rama = "โรงพยาบาลรามาธิบดี"
#chula = "โรงพยาบาลจุฬาลงกรณ์"
# These three hospitals are BIG tertiary care units and also be large medical schools.

#siph = "โรงพยาบาลศิริราช ปิยมหาราชการุณย์" "Only big private hospital that managed by NGO (Siriraj).
#b้hh = "โรงพยาบาลบำรุงราษฎร์" The most famous private hospital in Thailand.
#       (ref: https://healthmeth.wordpress.com/2019/08/30/5-top-best-private-hospital-thailand-2020/)
# To compare characteristics between these private hospitals. 

#chulabhorn = "โรงพยาบาลจุฬาภรณ์" Highest mean score from hospitals that has at least 100 comments.
#mit = "มิตรไมตรีคลินิกเวชกรรม สาขาวัดพระเงิน" The highest mean score from hospitals that has at least 30 comments.
#noparat = "โรงพยาบาลนพรัตนราชธานี" The lowest means score that has at least 30 and 100 comments.
# To compare between Good and Bad score hospitals

siriraj_v = full_data.loc[full_data['hospital'] == "โรงพยาบาลศิริราช"]
rama_v = full_data.loc[full_data['hospital'] == "โรงพยาบาลรามาธิบดี"]
chula_v = full_data.loc[full_data['hospital'] == "โรงพยาบาลจุฬาลงกรณ์"]
siph_v = full_data.loc[full_data['hospital'] == "โรงพยาบาลศิริราช ปิยมหาราชการุณย์"]
b้hh_v = full_data.loc[full_data['hospital'] == "โรงพยาบาลบำรุงราษฎร์"]
chulabhorn_v = full_data.loc[full_data['hospital'] == "โรงพยาบาลจุฬาภรณ์"]
mit_v = full_data.loc[full_data['hospital'] == "มิตรไมตรีคลินิกเวชกรรม สาขาวัดพระเงิน"]
noparat_v = full_data.loc[full_data['hospital'] == "โรงพยาบาลนพรัตนราชธานี"]

def score_graph(df):
    Score_count=df.groupby('score').count()
    Score_count['percentage'] = df['score'].value_counts(normalize=True) * 100
    plt.bar(Score_count.index.values, Score_count['comment'])
    plt.xlabel('Score')
    plt.ylabel('Number of Review')
    plt.show()
    print(Score_count[['comment','percentage']])

df_v.describe()
score_graph(full_data)

siriraj_v.describe()
score_graph(siriraj_v)

rama_v.describe()
score_graph(rama_v)

chula_v.describe()
score_graph(chula_v)

siph_v.describe()
score_graph(siph_v)

b้hh_v.describe()
score_graph(b้hh_v)

chulabhorn_v.describe()
score_graph(chulabhorn_v)

mit_v.describe()
score_graph(mit_v)

noparat_v.describe()
score_graph(noparat_v)

## Data preparation

df = full_data.copy()
df.shape
df.isnull().sum()
df.shape
df = df.drop_duplicates()
df.shape
# No duplicate or null data

# convert to sentiment type
# less than 3 is a negative sentiment = 0
df.loc[df['score'] < 3, 'sentiment'] = 0
# equal to 3 is a neutral sentiment, this group will not be used to build the model
df.loc[df['score'] == 3, 'sentiment'] = 'nan'
# more than 3 is a positive sentiment = 1
df.loc[df['score'] > 3, 'sentiment'] = 1

df=df.replace('nan',np.NaN)
df_neg = df[df['sentiment'] == 0]
df_pos = df[df['sentiment'] == 1]
df_full= pd.concat([df_neg, df_pos])

df_neg.shape
df_pos.shape
df_full.shape

siriraj_df = df_full.loc[df_full['hospital'] == "โรงพยาบาลศิริราช"]
rama_df = df_full.loc[df_full['hospital'] == "โรงพยาบาลรามาธิบดี"]
chula_df = df_full.loc[df_full['hospital'] == "โรงพยาบาลจุฬาลงกรณ์"]
siph_df = df_full.loc[df_full['hospital'] == "โรงพยาบาลศิริราช ปิยมหาราชการุณย์"]
b้hh_df = df_full.loc[df_full['hospital'] == "โรงพยาบาลบำรุงราษฎร์"]
chulabhorn_df = df_full.loc[df_full['hospital'] == "โรงพยาบาลจุฬาภรณ์"]
mit_df = df_full.loc[df_full['hospital'] == "มิตรไมตรีคลินิกเวชกรรม สาขาวัดพระเงิน"]
noparat_df = df_full.loc[df_full['hospital'] == "โรงพยาบาลนพรัตนราชธานี"]

def sentiment_graph(df):
    Sentiment_count = df.groupby('sentiment').count()
    Sentiment_count['percentage'] = df['sentiment'].value_counts(normalize=True)*100
    plt.bar(Sentiment_count.index.values, Sentiment_count['comment'])
    plt.xlabel('Review Sentiments')
    plt.ylabel('Number of Review')
    plt.show()
    print(Sentiment_count[['comment','percentage']])

sentiment_graph(df_full)
sentiment_graph(siriraj_df)
sentiment_graph(rama_df)
sentiment_graph(chula_df)
sentiment_graph(siph_df)
sentiment_graph(b้hh_df)
sentiment_graph(chulabhorn_df)
sentiment_graph(mit_df)
sentiment_graph(noparat_df)


## NLP processing

nlp = spacy.load('en_core_web_sm')

data = df_full.copy()
data = data.reset_index()
del data['index']
L  = []
for text in data['en']:
    doc = nlp(text)
    # including Tokenization
    # including Lower case      
    # Creating lemma words for this row
    lemmas = [token.lemma_ for token in doc]
    # Creating stop-words for this row
    stopwords = spacy.lang.en.stop_words.STOP_WORDS
    # Removing non-alphabetic characters
    # lemmatization
    # Removeing stop-words
    a_lemmas = [lemma for lemma in lemmas if lemma.isalpha() and lemma not in stopwords]
    b_lemmas = ' '.join(a_lemmas)
    L.append(b_lemmas)
    
df_L = pd.DataFrame(L, columns=['docc'])
data = pd.merge(data, df_L, left_index=True, right_index=True)
data.to_csv(r'D:/_RADS611 NLP/Assignment/honest_doc_full_post_nlp.csv')


nlp_data = pd.read_csv("D:/_RADS611 NLP/Assignment/honest_doc_full_post_nlp.csv")
nlp_data = nlp_data.reset_index()
del nlp_data['index']
nlp_data.head()

siriraj_nlp = nlp_data.loc[nlp_data['hospital'] == "โรงพยาบาลศิริราช"]
siriraj_nlp = siriraj_nlp.reset_index()
del siriraj_nlp['index']
siriraj_nlp.head()

rama_nlp = nlp_data.loc[nlp_data['hospital'] == "โรงพยาบาลรามาธิบดี"]
rama_nlp = rama_nlp.reset_index()
del rama_nlp['index']
rama_nlp.head()

chula_nlp = nlp_data.loc[nlp_data['hospital'] == "โรงพยาบาลจุฬาลงกรณ์"]
chula_nlp = chula_nlp.reset_index()
del chula_nlp['index']
chula_nlp.head()

siph_nlp = nlp_data.loc[nlp_data['hospital'] == "โรงพยาบาลศิริราช ปิยมหาราชการุณย์"]
siph_nlp = siph_nlp.reset_index()
del siph_nlp['index']
siph_nlp.head()

bhh_nlp = nlp_data.loc[nlp_data['hospital'] == "โรงพยาบาลบำรุงราษฎร์"]
bhh_nlp = bhh_nlp.reset_index()
del bhh_nlp['index']
bhh_nlp.head()

chulabhorn_nlp = nlp_data.loc[nlp_data['hospital'] == "โรงพยาบาลจุฬาภรณ์"]
chulabhorn_nlp = chulabhorn_nlp.reset_index()
del chulabhorn_nlp['index']
chulabhorn_nlp.head()

mit_nlp = nlp_data.loc[nlp_data['hospital'] == "มิตรไมตรีคลินิกเวชกรรม สาขาวัดพระเงิน"]
mit_nlp = mit_nlp.reset_index()
del mit_nlp['index']
mit_nlp.head()

noparat_nlp = nlp_data.loc[nlp_data['hospital'] == "โรงพยาบาลนพรัตนราชธานี"]
noparat_nlp = noparat_nlp.reset_index()
del noparat_nlp['index']
noparat_nlp.head()

## 3-gram Model

def tokenize(string):
    return re.findall(r'\w+', string.lower())

def count_ngrams(lines, min_length=2, max_length=4):
    """Iterate through given lines iterator (file object or list of
    lines) and return n-gram frequencies. The return value is a dict
    mapping the length of the n-gram to a collections.Counter
    object of n-gram tuple and number of times that n-gram occurred.
    Returned dict includes n-grams of length min_length to max_length.
    """
    lengths = range(min_length, max_length + 1)
    ngrams = {length: collections.Counter() for length in lengths}
    queue = collections.deque(maxlen=max_length)

    # Helper function to add n-grams at start of current queue to dict
    def add_queue():
        current = tuple(queue)
        for length in lengths:
            if len(current) >= length:
                ngrams[length][current[:length]] += 1

    # Loop through all lines and words and add n-grams to dict
    for line in lines:
        for word in tokenize(line):
            queue.append(word)
            if len(queue) >= max_length:
                add_queue()

    # Make sure we get the n-grams at the tail end of the queue
    while len(queue) > min_length:
        queue.popleft()
        add_queue()

    return ngrams


def print_most_frequent(ngrams, num=10):
    """Print num most common n-grams of each length in n-grams dict."""
    for n in sorted(ngrams):
        print('----- {} most common {}-grams -----'.format(num, n))
        for gram, count in ngrams[n].most_common(num):
            print('{0}: {1}'.format(' '.join(gram), count))
        print('')

def top_neg_pos(top_n, df):
    df_neg = df[df['sentiment'] == 0]
    df_pos = df[df['sentiment'] == 1]
    neg_gram = count_ngrams(df_neg['docc'], min_length=1, max_length=3)
    pos_gram = count_ngrams(df_pos['docc'], min_length=1, max_length=3)
    
    print(' ')
    print('Top ', str(top_n), 'negative group')
    print(' ')
    print_most_frequent(neg_gram, num=top_n)
    
    print(' ')
    print('Top ', str(top_n), 'positive group')
    print(' ')
    print_most_frequent(pos_gram, num=top_n)
    
    
top_neg_pos(10, siriraj_nlp)
top_neg_pos(10, rama_nlp)
top_neg_pos(10, chula_nlp)
top_neg_pos(10, siph_nlp)
top_neg_pos(10, bhh_nlp)
top_neg_pos(10, chulabhorn_nlp)
top_neg_pos(10, mit_nlp)
top_neg_pos(10, noparat_nlp)



## Word CLoud
def wordcloud(df):
    comment_words = '' 
    stopwords = set(STOPWORDS) 
    # iterate through the csv file 
    for val in df['docc']:  
        # typecaste each val to string 
        val = str(val) 
        # split the value 
        tokens = val.split() 
        # Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 
      
        comment_words += " ".join(tokens)+" "
  
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show() 


def cloud_neg_pos(df):
    df_neg = df[df['sentiment'] == 0]
    df_pos = df[df['sentiment'] == 1]
        
    print(' ')
    print('word cloud for negative group')
    print(' ')
    wordcloud(df_neg)
    
    print(' ')
    print('word cloud for positive group')
    print(' ')
    wordcloud(df_pos)


cloud_neg_pos(siriraj_nlp)
cloud_neg_pos(rama_nlp)
cloud_neg_pos(chula_nlp)
cloud_neg_pos(siph_nlp)
cloud_neg_pos(bhh_nlp)



## Radar chart
# Read the file
radar_data = pd.read_csv("D:/_RADS611 NLP/Assignment/6axis_csv2.csv")
cols = [ "Staff_neg", "Service_neg", "Waiting_neg", "Staff_pos", "Service_pos", "Waiting_pos"]
si_radar=[67,67,17,45,21,21]
rama_radar=[15,48,9,30,23,6]
chula_radar= [5,5,34,29,34,15]
siph_radar=[9,4,30,17,23,8]
bhh_radar=[43,29,54,19,52,9]

Attributes =list(radar_data)
AttNo = len(Attributes)


def create_radar(hospital, data):
    Attributes = [ "Staff_neg", "Service_neg", "Waiting_neg", "Staff_pos", "Service_pos", "Waiting_pos"]
    
    data += data [:1]
    
    angles = [n / 6 * 2 * pi for n in range(6)]
    angles += angles [:1]
    
    ax = plt.subplot(111, polar=True)

    plt.xticks(angles[:-1],Attributes)
    ax.plot(angles,data)
    ax.fill(angles, data, 'blue', alpha=0.1)

    ax.set_title(hospital)
    plt.show()

def compare_radar(hospital1, data, hospital2, data2):
    Attributes = [ "Staff_neg", "Service_neg", "Waiting_neg", "Staff_pos", "Service_pos", "Waiting_pos"]
    
    data += data [:1]
    data2 += data2 [:1]
    
    angles = [n / 6 * 2 * pi for n in range(6)]
    angles += angles [:1]
    
    angles2 = [n / 6 * 2 * pi for n in range(6)]
    angles2 += angles2 [:1]
    
    ax = plt.subplot(111, polar=True)
    ax = plt.subplot(111, polar=True)

    plt.xticks(angles[:-1],Attributes)

    ax.plot(angles,data)
    ax.fill(angles, data, 'teal', alpha=0.1)

    ax.plot(angles2,data2)
    ax.fill(angles2, data2, 'red', alpha=0.1)

    plt.figtext(0.2,0.9,hospital1,color="teal")
    plt.figtext(0.2,0.85,"v")
    plt.figtext(0.2,0.8,hospital2,color="red")
    plt.show()

def compare_3radar(hospital1, data, hospital2, data2, hospital3, data3):
    Attributes = [ "Staff_neg", "Service_neg", "Waiting_neg", "Staff_pos", "Service_pos", "Waiting_pos"]
    
    data += data [:1]
    data2 += data2 [:1]
    data3 += data3 [:1]
    
    angles = [n / 6 * 2 * pi for n in range(6)]
    angles += angles [:1]
    
    angles2 = [n / 6 * 2 * pi for n in range(6)]
    angles2 += angles2 [:1]
    
    angles3 = [n / 6 * 2 * pi for n in range(6)]
    angles3 += angles3 [:1]
    
    ax = plt.subplot(111, polar=True)
    ax = plt.subplot(111, polar=True)
    ax = plt.subplot(111, polar=True)
    
    plt.xticks(angles[:-1],Attributes)

    ax.plot(angles,data)
    ax.fill(angles, data, 'teal', alpha=0.1)

    ax.plot(angles2,data2)
    ax.fill(angles2, data2, 'red', alpha=0.1)

    ax.plot(angles3,data3)
    ax.fill(angles3, data3, 'green', alpha=0.1)

    plt.figtext(0.2,0.9,hospital1,color="teal")
    plt.figtext(0.2,0.85,"v")
    plt.figtext(0.2,0.8,hospital2,color="red")
    plt.figtext(0.2,0.75,"v")
    plt.figtext(0.2,0.7,hospital3,color="green")
    plt.show()


create_radar("Sririaj",si_radar)
create_radar("Rama",rama_radar)
create_radar("Chula",chula_radar)
create_radar("Siph",siph_radar)
create_radar("BHH",bhh_radar)

compare_radar("Siriraj",si_radar,"Rama",rama_radar)
compare_radar("Siriraj",si_radar,"Chula",chula_radar)
compare_radar("Chula",chula_radar,"Rama",rama_radar)
compare_radar("Siph",siph_radar,"BHH",bhh_radar)

compare_3radar("Siriraj",si_radar,"Rama",rama_radar, "Chula",chula_radar)


## LDA model using Bag-of-word and Tf-idf vectorization

docss= rama_df['en'].copy()
processed_docs= docss.copy()
processed_docs.head()

dictionary = gensim.corpora.Dictionary(processed_docs)

nltk.download('wordnet')
stemmer = SnowballStemmer('english')

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

processed_docs = docss.map(preprocess)
processed_docs.head()
processed_docs.shape

dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
    
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

for doc in corpus_tfidf:
    pprint(doc)
    break

#by Bag of words
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

#by tfidf
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)

for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))

