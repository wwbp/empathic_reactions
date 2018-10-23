import numpy as np
import pandas as pd
import scipy.stats as st
import sklearn.model_selection
import sys
import random
import os
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import KFold, ShuffleSplit
import keras.backend
from modeling.util import *


class Evaluator():
	def __init__(self, models):
		'''
		ARGS
			models			A dict mapping model string identifier to and 
							instance of the model class.
		'''
		self.models=models

	def __check_features__(self, features):
		'''
		Auxiliary method to check the formatting of the feature arrays.
		'''
		if isinstance(features,dict):
			print('Features are dictionary')
			if not len(features)==len(self.models):
				raise ValueError('The number of feature matrices must match '+\
					'the number of models')
		elif isinstance(features,np.ndarray):
				print('Features are ndarray')
				# builds up feature dictionary using copying by reference
				# all dictionary values now point to the same storage address
				features={key:features for key in self.models.keys()}
		else:
			raise ValueError('"features" must either be dict or ndarry!')
		return features

	def __cv__(self, features, labels, splits):
		'''
		Auxiliary method used by crossvalidate and repeat_crossvalidate
		'''
		results={key:pd.DataFrame(columns=labels.columns)\
			for key in list(self.models)}
		k=1
		for  train_index, test_index in KFold(n_splits=splits, shuffle=True).\
			split(labels):
			# print('...Clearing session...')
			print('........k={}'.format(k))
			keras.backend.clear_session() ### super-important if TF is used as backend. Otherwise memory leak
			# print('k='+str(k))
			labels_train=labels.iloc[train_index]
			labels_test=labels.iloc[test_index]
			# features_train=features.iloc[train_index]
			# features_test=features.iloc[test_index]
			

			for model_name in list(self.models):
				# print(model_name)
				 
				model=self.models[model_name]
				model.initialize() #resets the model.

				# retrieve model and split specific features
				features_train=np.take(features[model_name], 
					indices=train_index, axis=0)
				features_test=np.take(features[model_name], 
					indices=test_index, axis=0)


				model.fit(features_train, labels_train)
				preds=model.predict(features_test)
				# print(preds)
				# print(preds)
				preds=pd.DataFrame(preds, columns=labels.columns)
				results[model_name].loc[k]=eval(labels_test,preds)
			k+=1
		return results

	def repeat_crossvalidate(	self, 
								features, 
								labels, 
								k_splits, 
								iterations,
								outpath):
		'''
		Repeat k-fold crossvalidation n times. Saves average results for each
		iteration in tsv format.
		'''
		features=self.__check_features__(features)
		if not os.path.isdir(outpath):
			os.makedirs(outpath)
		for key, value in features.items():
			assert (len(value)==len(labels)), 'Features of model "{}" do not have the same length as the labels'.format(key)
		assert (k_splits>1), 'Crossvalidation makes no sense for k<2!'
		assert(iterations>1), 'Please use "crossvaliate" if you want only 1 iteration.'

		# setup df holding the average results of the individual CVs
		overall_results={key:pd.DataFrame(columns=labels.columns)\
			for key in list(self.models)}

		# do repeated crossvalidation
		for i in np.arange(1,iterations+1):
			print('....Iteration={}'.format(i))
			curr_results=self.__cv__(features, labels, k_splits)
			for model, df in curr_results.items():
				overall_results[model].loc[i]=df.mean(0)

		# compute mean and standard deviation and save results
		for model, df in overall_results.items():
			save_tsv(	df=average_results_df(df), 
						path='{}/{}.tsv'.format(outpath,model))

	def crossvalidate(self, features, labels, k_splits, outpath):
		'''
		Performs crossvalidation with each of the models given to this instance
		of the Evaluator class. The different models are tested on identical
		train/test splits which allows for using paired t-tests.

		ARGS
			features 		a dict mapping model names to np arrays. If only
							is given it assume all models will use the same 
							features.
		'''
		features=self.__check_features__(features)

		if not os.path.isdir(outpath):
			os.makedirs(outpath)

		# Checks that all feature matrices have the same length as the label matrix
		for key, value in features.items():
			assert (len(value)==len(labels)), 'Features of model "{}" do not have the same length as the labels'.format(key)
		assert (k_splits>1), 'Crossvalidation makes no sense for k<2!'

		# actual cross validation
		results=self.__cv__(features, labels, k_splits)

		# Averaging results
		for m_name in list(results):
			results[m_name]=average_results_df(results[m_name])
			save_tsv(df=results[m_name], path=outpath+'/'+m_name+'.tsv')

	


	def test_at_steps(self, features, labels, test_split, epochs, iterations, outpath):
		'''
		Repeatedly shuffles and splits data. Trains for a given number of
		epochs, test on held data after each epoch. Averages performance of
		each iteration. In the end, this results in smooth performance curves.

		ARGS
			features 		a dict mapping model names to np arrays. If only
							is given it assume all models will use the same 
							features.
		'''

		features=self.__check_features__(features)


		if not os.path.isdir(outpath):
			os.makedirs(outpath)


		#set up performance panels (one data frame per iteration)
		performances={key:{} for key in self.models.keys()}

		

		number_of_iteration=0
		for train_index, test_index in ShuffleSplit(n_splits=iterations, test_size=test_split).split(labels):
			print('...Clearing session...')
			keras.backend.clear_session() ### super-important if TF is used as backend. Otherwise memory leak
			number_of_iteration+=1
			for m_name, m in self.models.items():
				m.initialize()
				print(m_name)
				performances[m_name][number_of_iteration]=pd.DataFrame(columns=list(labels))
				print(performances[m_name][number_of_iteration])
			print('Now at iterateration {} of {}.'.format(number_of_iteration, iterations))
			
			# select iteration (but not model) specific labels
			labels_train=labels.iloc[train_index]
			labels_test=labels.iloc[test_index]

			# print(performances)
			
			for e in range(epochs):
				number_of_epoch=e+1
				print('\tepoch {} of {}'.format(number_of_epoch, epochs))
				for m_name, m in self.models.items():

					#retrieve model and iteration specific features
					features_train=np.take(features[m_name], indices=train_index, axis=0)
					features_test=np.take(features[m_name], indices=test_index, axis=0)


					m.train_epochs(features_train, labels_train, 1)
					preds=pd.DataFrame(m.predict(features_test), columns=list(labels))
					perf=eval(labels_test, preds)
					performances[m_name][number_of_iteration].loc[number_of_epoch]=perf


		#average and save results
		for m in self.models.keys():
			panel=pd.Panel(performances[m])
			mean_perf=panel.mean(axis=0)
			save_tsv(mean_perf, outpath+'/{}.tsv'.format(m))