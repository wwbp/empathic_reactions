import heapq
import io
import numpy as np
from numpy.linalg import norm
import zipfile
# from framework.representations.matrix_serializer import load_vocabulary


class Embedding:
    def __init__(self, matrix, vocabulary, word2index, normalize):
        '''
        Args:
            matrix          A numpy array, words associated with rows
            vocabulary      List of strings
            word2index      Dictionary mapping word to its index in 
                            "vocabulary".
            normalized      Boolean
        '''
        self.m=matrix
        print(self.m.shape)
        self.normalized=normalize
        if normalize:
            self.normalize()
        self.dim=self.m.shape[1]
        self.wi=word2index
        self.iw=vocabulary


    def normalize(self):
        norm = np.sqrt(np.sum(self.m * self.m, axis=1))
        self.m = self.m / norm[:, np.newaxis]
        self.normalized=True

    def represent(self, w):
        if w in self.wi:
            return self.m[self.wi[w], :]
        else:
            return np.zeros(self.dim)

    def similarity(self, w1, w2):
        if self.normalized:
            return self.represent(w1).dot(self.represent(w2))
        else:
            e1=self.represent(w1)
            e2=self.represent(w2)
            return e1.dot(e2)/(norm(e1)*norm(e2))

    def closest(self, w, n=10):
        """
        Assumes the vectors have been normalized.
        """
        scores = self.m.dot(self.represent(w))
        return heapq.nlargest(n, zip(scores, self.iw))

    def subsample(self, kept_words):
        '''
        Return new embeddings model where only the "kept_words" remain
        represented.
        '''
        #keep all words present in both.
        # new_iw=list(set(kept_words).intersection(self.iw))
        new_iw=list(set(kept_words)) #it might be that my new embeddings space 
                                     # should contain words that are acutally
                                     # not in the original one...
        # print(kept_words)
        new_wi={new_iw[i]:i for i in range(len(new_iw))}
        new_m=np.zeros(shape=[len(new_iw), self.dim])
        # print(new_m)
        for i in range(len(new_iw)):
            new_m[i,:]=self.represent(new_iw[i])
        new_embeddings=Embedding(   matrix=new_m,
                                    vocabulary=new_iw,
                                    word2index=new_wi,
                                    normalize=self.normalized)
        # print('New embeddings: ',new_embeddings.m)
        # print(' old embeddings: \n', self.m)
        # print('new embeddings: \n', new_embeddings.m)
        return new_embeddings
    

    @classmethod
    def from_word2vec_bin(cls, path, vocab_limit=None, normalize=False):
        from gensim.models.keyedvectors import KeyedVectors
        model = KeyedVectors.load_word2vec_format(path, binary=True, limit=vocab_limit)
        iw=list(model.vocab)
        dim=len(model[iw[0]])
        m=np.zeros([len(iw),dim])
        for i in range(len(iw)):
            m[i,]=model[iw[i]]
        wi={iw[i]:i for i in range(len(iw))}
        return cls( matrix=m,
                    vocabulary=iw,
                    word2index=wi,
                    normalize=normalize)

    @classmethod 
    def from_raw_format(    cls,
                            path,
                            vocab_limit=-1,
                            normalize=False,
                            delim=' '):
        '''
        Method to read embeddings from textfile. One word per line. Word is first
        entry and is seperated by "delim". The vector components are also 
        separated by "delim".
        '''
        with io.open(path,'r', encoding='utf8') as f:
            vectors=[]
            wi={}
            iw=[]
            line='start'
            count=0
            # READ FIRST LINE TO FIND OUT DIMENSIONALITY
            line=f.readline().strip()
            parts=line.split(delim)
            dim=len(parts)-1
            word=' '.join(parts[:-dim])
            vec=[float(x) for x in parts[-dim:]]
            iw+=[word]
            wi[word]=count
            vectors.append(vec)
            count+=1
            # READ ALL THE OTHER LINES
            while count<vocab_limit or vocab_limit==-1:
                line=f.readline().strip()
                # STOPS ITERATING END END OF FILE IS REACHED
                if line=='':
                    break
                parts=line.split(delim)
                word=' '.join(parts[:-dim])
                vec=[float(x) for x in parts[-dim:]]
                iw+=[word]
                wi[word]=count
                vectors.append(vec)
                count+=1
            return cls( matrix=np.array(vectors),
                    vocabulary=iw,
                    word2index=wi,
                    normalize=normalize)

    @classmethod
    def from_fasttext_vec(  cls, 
                            path,
                            zipped=False,
                            file=None, 
                            vocab_limit=None, 
                            normalize=False):
        '''
        Method to read the plain text format of fasttext (usually ending with
        .vec).

        First line consists of <vocab_size><blank><dimensions>.
        '''
        if zipped==True and file is None:
            raise ValueError('You are trying to load a file withing a ZIP '+\
                'but have not indicated the name of this file.')
        
        #vectors=[]
        wi={}
        iw=[]
        m=None

        if not zipped:
            with io.open(path,'r', encoding='utf8') as f:
                # for line in f.readlines(vocab_limit+1)[1:]:
                #   count+=1
                
                first_line=f.readline().split() 
                vocab_size=int(first_line[0])
                dim=int(first_line[1])
                if vocab_limit is None:
                    vocab_limit=vocab_size
                m=np.zeros([vocab_limit,dim])
                for count in range(vocab_limit):
                    if count%100000==0:
                        print(count)
                    line=f.readline().strip()
                    # print(count)
                    parts=line.split()
                    word=' '.join(parts[:-dim])
                    # print(len(parts))
                    # print(word)
                    #vec=[float(x) for x in parts[-dim:]]
                    m[count,:]=[float(x) for x in parts[-dim:]]
                    iw+=[word]
                    wi[word]=count
                    #vectors.append(vec)
                    # print(vec)
        elif zipped:
            with zipfile.ZipFile(path) as z:
                with z.open(file) as f:
                    
                    # for line in f.readlines(vocab_limit+1)[1:]:
                    #   count+=1
                    
                    first_line=f.readline().split() 
                    vocab_size=int(first_line[0])
                    dim=int(first_line[1])
                    if vocab_limit is None:
                        vocab_limit=vocab_size
                    m=np.zeros([vocab_limit,dim])
                    for count in range(vocab_limit):
                        if count%100000==0:
                            print(count)
                        line=f.readline().decode('utf8').strip()
                        # print(count)
                        parts=line.split()
                        word=' '.join(parts[:-dim])
                        # print(len(parts))
                        # print(word)
                        #vec=[float(x) for x in parts[-dim:]]
                        m[count,:]=[float(x) for x in parts[-dim:]]
                        iw+=[word]
                        wi[word]=count
                        #vectors.append(vec)
        # print(vectors[:5])
        return cls( matrix=m,#np.array(vectors),
                    vocabulary=iw,
                    word2index=wi,
                    normalize=normalize)
       



class FastTextEmbedding(Embedding):
	'''
	Reads embeddings from plain text file.
	vocab_limit.........The number of words to read.
	'''
	def __init__(self, path, vocab_limit, dims=300, zipped=False,
		file=None, sep=' ', normalize=False, skip_lines=1):
		if zipped==True:
			assert file is not None, 'Indicate file name within the zip archive!'
		if zipped:
			with zipfile.ZipFile(path) as z:
				with z.open(file) as f:
					vectors=[]
					self.wi={}
					self.iw=[]
					# for line in f.readlines(vocab_limit+1)[1:]:
					# 	count+=1
					for __ in range(skip_lines):
						f.readline() # first line includes only dim number and 
									 # vocab size
					for count in range(vocab_limit):
						line=f.readline().decode('utf8').strip()
						# print(count)
						parts=line.split()
						word=' '.join(parts[:-dims])
						# print(len(parts))
						# print(word)
						vec=[float(x) for x in parts[-dims:]]
						self.iw+=[word]
						self.wi[word]=count
						vectors.append(vec)
						# if count==512:
						# 	print(parts)
						# 	print(vec)
					# while count<vocab_limit or vocab_limit is None:
					# 	try:
					# 		line = f.__next__()
					# 		parts=line.split(sep)
					# 		self.iw+=[parts[0]]
					# 		self.wi[parts[0]]=count
					# 		vectors.append(parts[1:])
					# 	except StopIteration:
					# 		break
					self.m=np.array(vectors)
					if normalize:
						self.normalize()
					normalized=normalize
					self.dim = self.m.shape[1]
					print(self.m)
					# print(self.iw[0], self.m[0])
					# print(self.m.shape)


		else:
			raise NotImplementedError

