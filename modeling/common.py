import numpy as np
import pandas as pd


from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, Concatenate, Bidirectional, Conv1D, MaxPool1D,Conv2D, MaxPool2D, Reshape, Flatten, AveragePooling1D
from keras.layers import CuDNNGRU as GRU, CuDNNLSTM as LSTM
from keras.optimizers import Adam
from keras import regularizers

from modeling import feature_extraction as fe
import util

from sklearn.linear_model import ElasticNet, LinearRegression, RidgeCV
from sklearn.base import clone

from modeling.embedding import Embedding

import os
import numpy as np
import pandas as pd
import constants as cs
from io import StringIO
import re


def tokenize(string):
	return re.split(r'\b',string)


def get_google_sgns(vocab_limit=None):
	return Embedding.from_word2vec_bin(
			path=cs.google_news_embeddings,
			vocab_limit=vocab_limit)

def get_facebook_fasttext_wikipedia(language, vocab_limit=None):
	return Embedding.from_fasttext_vec(
			path=cs.facebook_fasttext_wikipedia[language],
			vocab_limit=vocab_limit)

def get_facebook_fasttext_common_crawl(vocab_limit=None):
	return Embedding.from_fasttext_vec(
			path=cs.facebook_fasttext_common_crawl,
			vocab_limit=vocab_limit,
			zipped=True,
			file='crawl-300d-2M.vec')







def get_message_data():
	return pd.read_csv(cs.messages, index_col=0)

def get_article_data():
	return pd.read_csv(cs.articles, index_col=0)


TIMESTEPS=200


def get_rnn(rnn_type,
			input_shape, 
			output_num,  
			rnn_units, 
			dense_units, 
			dropout_embedding, 
			dropout_recurrent, 
			dropout_dense, 
			learning_rate, 
			bidirectional):
	'''
	rnn_type			'gru' or 'lstm'
	'''
	RNN_Type={'gru':GRU, 'lstm':LSTM}[rnn_type]
	model=None
	layers=[]

	layers.append(Input(shape=(input_shape[0],input_shape[1],)))
	layers.append(Dropout(rate=dropout_embedding)(layers[-1]))
	# layers.append(GRU(100, dropout=.2, return_sequences=True)(layers[-1]))
	if bidirectional:
		layers.append(Bidirectional(RNN_Type(rnn_units))(layers[-1]))
	else:
		layers.append(RNN_Type(rnn_units)(layers[-1]))
	layers.append(Dropout(rate=dropout_recurrent)(layers[-1]))
	layers.append(Dense(dense_units)(layers[-1]))
	layers.append(Activation('relu')(layers[-1]))
	layers.append(Dropout(rate=dropout_dense)(layers[-1]))
	layers.append(Dense(output_num)(layers[-1]))
	model = Model(inputs=layers[0], outputs=layers[-1])
	optimizer=Adam(lr=learning_rate)
	model.compile(loss='mse', optimizer=optimizer, metrics=[])
	model=model

	return model



def get_ffn(units, dropout_hidden, dropout_embedding, learning_rate, problem='regression'):
	l2_strength=.001

	if problem not in ['regression', 'classification']:
		raise ValueError
	layers=[]
	layers.append(Input(shape=(units[0],)))
	layers.append(Dropout(rate=dropout_embedding)(layers[-1]))
	for i, curr_units in enumerate(units[1:-1]):
		layers.append(Dense(curr_units, activation='relu', kernel_regularizer=regularizers.l2(l2_strength))(layers[-1]))
		layers.append(Dropout(rate=dropout_hidden)(layers[-1]))
	
	if problem=='classification':
		layers.append(Dense(units[-1], activation='sigmoid', kernel_regularizer=regularizers.l2(l2_strength))(layers[-1]))
	if problem=='regression':
		layers.append(Dense(units[-1], kernel_regularizer=regularizers.l2(l2_strength))(layers[-1]))
	model=Model(inputs=layers[0], outputs=layers[-1])
	optimizer=Adam(lr=learning_rate)#, decay=1e-4)
	if problem=='classification':
		model.compile(loss='binary_crossentropy', optimizer=optimizer)
	if problem=='regression':
		model.compile(loss='mse', optimizer=optimizer)
	return model

def get_cnn(input_shape, num_outputs, num_filters, learning_rate, dropout_conv, problem):
	
	if problem not in ['regression', 'classification']:
		raise ValueError
	
	# loosely based on https://github.com/bhaveshoswal/CNN-text-classification-keras/blob/master/model.py

	# filter_sizes=[3,4,5]
	embedding_dim=input_shape[1]
	sequence_length=input_shape[0]

	

	l2_strength=.001

	inputs = Input(shape=input_shape)
	inputs_drop = Dropout(dropout_conv)(inputs)

	filter_size=1
	conv_1=Conv1D(filters=num_filters, kernel_size=filter_size, strides=1, activation='relu', kernel_regularizer=regularizers.l2(l2_strength))(inputs_drop)
	pool_1=AveragePooling1D(pool_size=input_shape[0]-filter_size+1, strides=1)(conv_1)
	pool_drop_1=Dropout(dropout_conv)(pool_1)

	filter_size=2
	conv_2=Conv1D(filters=num_filters, kernel_size=filter_size, strides=1, activation='relu', kernel_regularizer=regularizers.l2(l2_strength))(inputs_drop)
	pool_2=AveragePooling1D(pool_size=input_shape[0]-filter_size+1, strides=1)(conv_2)
	pool_drop_2=Dropout(dropout_conv)(pool_2)
	
	filter_size=3
	conv_3=Conv1D(filters=num_filters, kernel_size=filter_size, strides=1, activation='relu', kernel_regularizer=regularizers.l2(l2_strength))(inputs_drop)
	pool_3=AveragePooling1D(pool_size=input_shape[0]-filter_size+1, strides=1)(conv_3)
	pool_drop_3=Dropout(dropout_conv)(pool_3)

	# filter_size=4
	# conv_4=Conv1D(filters=256, kernel_size=filter_size, strides=1, activation='relu', kernel_regularizer=regularizers.l2(l2_strength))(inputs_drop)
	# maxpool_4=MaxPool1D(pool_size=input_shape[0]-filter_size+1, strides=1)(conv_4)
	# pool_drop_4=Dropout(.2)(maxpool_4)

	# filter_size=5
	# conv_5=Conv1D(filters=256, kernel_size=filter_size, strides=1, activation='relu', kernel_regularizer=regularizers.l2(l2_strength))(inputs_drop)
	# maxpool_5=MaxPool1D(pool_size=input_shape[0]-filter_size+1, strides=1)(conv_5)
	# pool_drop_5=Dropout(.2)(maxpool_5)

	concatenated=Concatenate(axis=1)([pool_drop_1, pool_drop_2, pool_drop_3])

	dense = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(l2_strength))(Flatten()(concatenated))
	dense_drop = Dropout(.5)(dense)
	
	if problem=='classification':
		output = Dense(units=num_outputs, activation='sigmoid', kernel_regularizer=regularizers.l2(l2_strength))(dense_drop)
	if problem=='regression':
		output = Dense(units=num_outputs, activation=None, kernel_regularizer=regularizers.l2(l2_strength))(dense_drop)
	# this creates a model that includes
	model = Model(inputs=inputs, outputs=output)
	optimizer=Adam(lr=learning_rate)

	if problem=='regression':
		model.compile(loss='mse', optimizer=optimizer)
	if problem=='classification':
		model.compile(loss='binary_crossentropy', optimizer=optimizer)
	return model



def get_cnn_lstm(input_shape, num_outputs, learning_rate, problem):

	l2_strength=.001


	if problem not in ['regression', 'classification']:
		raise ValueError
	filter_size=3
	embedding_dim=input_shape[1]
	sequence_length=input_shape[0]

	inputs = Input(shape=input_shape)
	inputs_drop = Dropout(.2)(inputs)
	conv=Conv1D(filters=128, kernel_size=3, strides=1, activation='relu', kernel_regularizer=regularizers.l2(l2_strength))(inputs_drop)
	maxpool=AveragePooling1D(pool_size=2, strides=1)(conv)
	pool_drop=Dropout(.2)(maxpool)


	lstm=LSTM(256, kernel_regularizer=regularizers.l2(l2_strength))(pool_drop)
	lstm_drop=Dropout(.5)(lstm)
	dense=Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_strength))(lstm_drop)
	dense_drop=Dropout(.5)(dense)
	if problem=='regression':
		output=Dense(units=num_outputs, activation=None, kernel_regularizer=regularizers.l2(l2_strength))(dense_drop)
	if problem=='classification':
		output=Dense(units=num_outputs, activation='sigmoid', kernel_regularizer=regularizers.l2(l2_strength))(dense_drop)
	model = Model(inputs=inputs, outputs=output)
	optimizer=Adam(lr=learning_rate)
	if problem=='classification':
		model.compile(loss='binary_crossentropy', optimizer=optimizer)
	if problem=='regression':
		model.compile(loss='mse', optimizer=optimizer)
	return model



