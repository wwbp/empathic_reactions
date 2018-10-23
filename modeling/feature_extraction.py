#from nltk.tokenize import word_tokenize as tokenize
from nltk.tokenize import wordpunct_tokenize as tokenize
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def embedding_centroid(raw_data, embs):
	'''
	Turns a list of linguistic units (sentences, texts,...) into a 2d tensor,
	such that each unit is represented a vector which constitutes the
	centroid of the embedding vectors of the individual words.
	'''
	raw_data=list(raw_data)
	out_matrix=np.zeros([len(raw_data),embs.dim])
	for i in range(len(raw_data)):
		# print(i)
		sentences=[str.lower() for str in tokenize(raw_data[i])]
		# print(sent)
		features=[embs.represent(str) for str in sentences]
		features=np.stack(features, axis=0)
		# print(features)
		out_matrix[i,:]=features.mean(axis=0)
	return out_matrix


def embedding_matrix(raw_data, embs, fixed_len):
	'''
	Expects list of strings.
	Turns a list of linguistic units (sentences, texts,...) into a 3d tensor,
	such that each unit is represented by a matrix of concatenated embeddings
	of the words in this unit.
	
	'''
	matrices = []
	for sent in raw_data:
		sent=[str.lower() for str in tokenize(sent)]
		# print(sent)
		features=[embs.represent(str) for str in sent]
		# now pads the left margin
		zeros=np.zeros((fixed_len, embs.dim))
		i=1
		while i <= fixed_len and i <= len(features):
			zeros[-i,:]=features[-i]
			i+=1


		# features=np.stack(features, axis=0)
		# # print(features)
		# # print(features.shape)
		# features=__pad_array__(features, fixed_len)
		# print(zeros)

		matrices.append(zeros)
	return np.stack(matrices, axis=0)

def bag_of_ngrams(raw_data,bool_features):
	vectorizer=CountVectorizer(input='content', lowercase=True,
		tokenizer=tokenize, min_df=2, binary=bool_features, ngram_range=(1,3))
	features=vectorizer.fit_transform(raw_data).toarray()
	# print(vectorizer.vocabulary_.keys())
	return features



def __pad_array__(array, fixed_len):
	'''
	transforms 2d np array of shape (x,y) into shape
	(fixed_len,y)
	https://stackoverflow.com/questions/9251635/python-resize-an-existing-array-and-fill-with-zeros
	'''
	
	# print(array[:fixed_len, :array.shape[1]])

	if array.shape[0]>fixed_len:
		### trim array
		return array[:fixed_len, :]
	elif array.shape[0]<fixed_len:
		zeros=np.zeros((fixed_len,array.shape[1]), dtype=np.float32)
		zeros[:array.shape[0], :array.shape[1]]=array
		return zeros
	else:
		return array
