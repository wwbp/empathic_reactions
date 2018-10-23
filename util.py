import constants  as cs
import pandas as pd 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
import scipy.stats as st

def no_zeros_formatter(x, decimals=3):
	formatter='{'+':.{}f'.format(decimals)+'}'
	return formatter.format(round(x,decimals)).lstrip('0').replace('-0.', '-.')


# Median split
def median_split(series):
	return series.apply(lambda x: 0 if x <= series.median() else 1)



### LOADING DATA
def get_articles():
	ratings=pd.read_csv(cs.articles, index_col=0)
	raw=pd.read_csv(cs.articles_data, index_col=0)
	return ratings.merge(raw, how='outer', left_index=True, right_index=True)

def get_messages():
	return pd.read_csv(cs.messages)


def plot_histogram(series, outfile, bins=100):
	fig=plt.figure()
	series.hist(xrot=45, bins=bins)
	# counts=column.value_counts()
	# print(counts
	plt.tight_layout()
	fig.savefig(outfile)
	plt.close()

def train_dev_test_split(df):
	train, dev, test = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df)), int(.9*len(df))])
	return train, dev, test


def split_half_reliability(df, seed, iterations=100):
	#splits items in columns repeatedly into two groups and computes correlation
	#between group averages
	cols=list(df)
	half_index=round(len(cols)/2.)
	results=[]
	random.seed(seed)
	for i in range(iterations):
		random.shuffle(cols)
		group_1=cols[:half_index]
		group_2=cols[half_index:]
		avg_1=df[group_1].mean(1)
		avg_2=df[group_2].mean(1)
		results.append(st.pearsonr(avg_1,avg_2)[0])
	return(np.mean(results))





