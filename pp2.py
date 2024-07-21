import string
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import os
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from apyori import apriori
from tqdm import tqdm
from collections import Counter

DIR_DATA = './20news-bydate-train'

list_results = []
for target in tqdm(os.listdir(DIR_DATA)):
    for text in os.listdir(f'{DIR_DATA}/{target}'):
        with open(f'{DIR_DATA}/{target}/{text}', 'r') as f:
            list_results.append( {'text': f.read()})
			

df_result = pd.DataFrame(list_results)

re_url = re.compile(r'(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
re_email = re.compile('(?:[a-z0-9!#$%&\'+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])')

#Removing the headers at every file
def clean_header(text):
    text = re.sub(r'(From:\s+[^\n]+\n)', '', text)
    text = re.sub(r'(Subject:[^\n]+\n)', '', text)
    text = re.sub(r'(([\sA-Za-z0-9\-]+)?[A|a]rchive-name:[^\n]+\n)', '', text)
    text = re.sub(r'(Last-modified:[^\n]+\n)', '', text)
    text = re.sub(r'(Version:[^\n]+\n)', '', text)
    return text

df_result['text_cleaned'] = df_result['text'].apply(clean_header)

def clean_text(text):
    text = text.lower()
    text = text.strip()
    text = re.sub(re_url, '', text)
    text = re.sub(re_email, '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'(\d+)', ' ', text)
    text = re.sub(r'(\s+)', ' ', text)
    return text

df_result['text_cleaned'] = df_result['text_cleaned'].apply(clean_text)
df_result.drop('text', inplace=True, axis=1)

#Delete Stopwords
stop_words = stopwords.words('english')

df_result['text_cleaned'] = df_result['text_cleaned'].str.split() \
    .apply(lambda x: ' '.join([word for word in x if word not in stop_words]))

#Stemming	
stemmer = PorterStemmer()

df_result['text_cleaned'] = df_result['text_cleaned'].str.split() \
    .apply(lambda x: ' '.join([stemmer.stem(word) for word in x]))

#Get 10 most frequent words
def get_ten_most_frequent_words(text):
    # split the text into words
    words = text.split()
    word_counts = Counter(words)
    # sort the words by frequency in descending order
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True) 
    return [word for word, count in sorted_word_counts][0:10]

for i in df_result.index:
    df_result['text_cleaned'][i] = get_ten_most_frequent_words(df_result['text_cleaned'][i])

#Save the words in a csv file
df_result.to_csv('Results.csv')	
	
#apriori
transactions = []
for i in df_result.index:
    transactions.append(df_result['text_cleaned'][i])

association_rules = apriori(transactions, min_support=0.01, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)

#Print the rules in the console
print(association_results)

#Save the Association Rules in a csv
Rules = pd.DataFrame(association_results)
Rules.to_csv('Association_Rules.csv')
	


