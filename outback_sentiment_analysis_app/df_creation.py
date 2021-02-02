import pandas as pd 
import streamlit as st

df_original = pd.read_csv('web_scrape_tripadvisor_final.csv')


def get_bubble(text):
    if text == 'bubble_50':
        return 5
    elif text == 'bubble_40':
        return 4
    elif text == 'bubble_30':
        return 3
    elif text == 'bubble_20':
        return 2
    else:
        return 1


lista_months = 'janeiro fevereiro março abril maio junho julho agosto setembro outubro novembro dezembro'.split()

def translate(month):
    if month == lista_months[0]:
        return '01'
    elif month == lista_months[1]:
        return '02'
    elif month == lista_months[2]:
        return '03'
    elif month == lista_months[3]:
        return '04'
    elif month == lista_months[4]:
        return '05'
    elif month == lista_months[5]:
        return '06'
    elif month == lista_months[6]:
        return '07'
    elif month == lista_months[7]:
        return '08'
    elif month == lista_months[8]:
        return '09'
    elif month == lista_months[9]:
        return '10'
    elif month == lista_months[10]:
        return '11'
    elif month == lista_months[11]:
        return '12'


def final_semana(col):
    if col == 5 or col == 6:
        return '1'
    else:
        return '0'


df_time = (
    df_original
    .assign(day = df_original.date.str.split(" ",expand=True,)[0])
    .assign(p1 = df_original.date.str.split(" ",expand=True,)[1])
    .assign(month = df_original.date.str.split(" ",expand=True,)[2].apply(translate))
    .assign(p2 = df_original.date.str.split(" ",expand=True,)[3])
    .assign(year = df_original.date.str.split(" ",expand=True,)[4])
    .drop(columns = ['p1', 'p2'], axis = 1)
    .copy()
)

df_time = df_time\
    .assign(final_date = df_time[['day', 'month','year']].apply(lambda row: '-'.join(row.values.astype(str)), axis=1))
df_time.final_date = pd.to_datetime(df_time.final_date)
df_time.drop(columns = ['date', 'Unnamed: 0'], axis = 1, inplace = True)
df_time.loc[:, 'day_of_week'] = df_time.final_date.dt.dayofweek
df_time.loc[:,'name_day'] = df_time.final_date.dt.day_name()
df_time.loc[:, 'quarter'] = df_time.final_date.dt.quarter
df_time.loc[:, 'is_weekend'] = df_time['day_of_week'].apply(final_semana)

df_time['rating'] = df_time['rating'].apply(get_bubble)

df_time.to_csv('dataframe_time.csv')

df = df_original.drop('date', axis = 1)

import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


def clean_text(text):
    stop_words = set(stopwords.words('portuguese'))

    t = re.sub(r'http\s+', '', str(text))
    t = re.sub('[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]', ' ', str(t))
    t = str(t).lower()
    t = word_tokenize(t)
    t = [word for word in t if word not in stop_words]
    t = [word for word in t if len(word) > 2]
    t =  ' '.join(t)
    
    
    return t

df['final_text'] = df['text'].apply(clean_text)
df['final_title'] = df['title'].apply(clean_text)
df = df.assign(title_text = lambda row: row['final_text'] + ' ' + row['final_title'])
df['rating'] = df['rating'].apply(get_bubble)


def transform_rating(rating):
    if rating == 5 or rating == 4:
        return 1
    else:
        return 0

df['rating'] = df['rating'].apply(transform_rating)


df['year'] = df_time['year']
df.to_csv('Dataframe_model.csv')