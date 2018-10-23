import os
from os.path import join as jn
import pandas as pd


# Paths
root=os.environ['EMPATHY_PROJECT_ROOT']
responses_raw=jn(root,'data','responses','data','qualtrics_download_2018-05-16-1044.csv')
responses=root+'/data/responses/data/responses.csv'
responses_to_exclude=root+'/data/responses/data/exclude_responses.txt'
messages=root+'/data/responses/data/messages.csv'
articles=jn(root, 'data', 'responses', 'data', 'articles.csv')
articles_raw=root+'/data/stimulus/data/empathynewsarticles_all.txt' # original source of articles provided by Joao 
articles_data=root+'/data/stimulus/data/articles_raw_data.csv'
articles_categories=root+'/data/stimulus/data/article_categories.csv'
articles_full=root+'/data/articles_full_data.csv'




# word vectors
vectors=os.environ['VECTORS']
facebook_fasttext_common_crawl=jn(vectors,'crawl-300d-2M.vec.zip')
google_news_embeddings='/data1/embeddings/eng/GoogleNews-vectors-negative300.bin'
