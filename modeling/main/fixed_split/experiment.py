import pandas as pd
import util
import modeling.feature_extraction as fe
from modeling import common
from scipy import stats as st
from sklearn import metrics
import numpy as np
import keras
from sklearn.linear_model import RidgeCV, LogisticRegressionCV

## extra imports to set GPU options
import tensorflow as tf
from keras import backend as k
 
###################################
# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
 
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
###################################


#######		Setting up data		########
train, dev, test=util.train_dev_test_split(util.get_messages())
train=pd.concat([train,dev])
# print(train.head())
########################################

results_df=pd.DataFrame(
	index=['cnn', 'ffn', 'ridge', 'maxent'], 
	columns=['empathy', 'empathy_bin', 'distress', 'distress_bin'])


embs=common.get_facebook_fasttext_common_crawl(vocab_limit=int(100e3))

TARGETS=['empathy', 'distress']

MODELS_REGRESSION={
	'cnn':lambda:common.get_cnn(
							input_shape=[common.TIMESTEPS,300], 
							num_outputs=1, 
							num_filters=100, 
							learning_rate=1e-3,
							dropout_conv=.5, 
							problem='regression'),

	'ffn': lambda:	common.get_ffn(
							units=[300,256, 128,1], 
							dropout_hidden=.5,
							dropout_embedding=.2, 
							learning_rate=1e-3,
							problem='regression'),

	'ridge': lambda: RidgeCV(
							alphas=[1, 5e-1, 1e-1,5e-2, 1e-2, 5e-3, 1e-3,5e-4, 1e-4])
}

MODELS_CLASSIFICATION={
	'cnn':lambda:common.get_cnn(
							input_shape=[common.TIMESTEPS,300], 
							num_outputs=1, 
							num_filters=100, 
							learning_rate=1e-3,
							dropout_conv=.5, 
							problem='classification'),

	'ffn': lambda:	common.get_ffn(
							units=[300,256, 128,1], 
							dropout_hidden=.5,
							dropout_embedding=.2, 
							learning_rate=1e-3,
							problem='classification'),

	'maxent': lambda: LogisticRegressionCV(Cs=20)
}

PROBLEMS=['classification', 'regression']

MODELS={
	'classification':MODELS_CLASSIFICATION,
	'regression':MODELS_REGRESSION
}

features_train_centroid=fe.embedding_centroid(train.essay, embs)
features_train_matrix=fe.embedding_matrix(train.essay, embs, common.TIMESTEPS)

features_test_centroid=fe.embedding_centroid(test.essay, embs)
features_test_matrix=fe.embedding_matrix(test.essay, embs, common.TIMESTEPS)

LABELS={
	'empathy':{'classification':'empathy_bin', 'regression':'empathy'},
	'distress':{'classification':'distress_bin', 'regression':'distress'}
}

def f1_score(true, pred):
	pred=np.where(pred.flatten() >.5 ,1,0)
	result=metrics.precision_recall_fscore_support(
		y_true=true, y_pred=pred, average='micro')
	return result[2]

def correlation(true, pred):
	pred=pred.flatten()
	result=st.pearsonr(true,pred)
	return result[0]

SCORE={
	'classification': f1_score,
	'regression':correlation

}

early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=20,
                              verbose=0, mode='auto')

for target in TARGETS:
	print(target)
	for problem in PROBLEMS:
		print(problem)

		labels_train=train[LABELS[target][problem]]
		labels_test=test[LABELS[target][problem]]

		for model_name, model_fun in MODELS[problem].items():
			print(model_name)
			model=model_fun()


			#	TRAINING
			if model_name=='cnn':
				model.fit(	features_train_matrix, 
							labels_train,
							epochs=200, 
							batch_size=32, 
							validation_split=.1, 
							callbacks=[early_stopping])

			elif model_name=='ffn':
				model.fit(	features_train_centroid,
							labels_train,
							epochs=200, 
							validation_split=.1, 
							batch_size=32, 
							callbacks=[early_stopping])

			elif model_name=='ridge':
				model.fit(	features_train_centroid,
							labels_train)

			elif model_name=='maxent':
				model.fit(	features_train_centroid,
							labels_train)

			else:
				raise ValueError('Unkown model name encountered.')

			#	PREDICTION
			if model_name=='cnn':
				pred=model.predict(features_test_matrix)
			else:
				pred=model.predict(features_test_centroid)

			#	SCORING
			result=SCORE[problem](true=labels_test, pred=pred)

			#	RECORD
			row=model_name
			column=LABELS[target][problem]
			results_df.loc[row,column]=result
			print(results_df)

results_df.to_csv('results.tsv', sep='\t')





