import os
import numpy as np
import pandas as pd
import tensorflow as tf
import string
import json
import glob
import time
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import  Dense, LSTM, Dropout, Embedding, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import add
import pickle


#%%

class Flicker_Dataset:
    
    def __init__(self):
        self.file_name = 'Flicker_Text/Flickr8k.token.txt'
        self.images_path = 'Flicker8k_Dataset/'
        self.vocabulary = set() 

    def clean_description(self,caption):
     	caption = [ch for ch in caption if ch not in string.punctuation]
     	caption = ''.join(caption)
     	caption = caption.split(' ')
     	caption = [word.lower() for word in caption if len(word)>1 and word.isalpha()]
     	self.vocabulary.update(caption)
     	caption = ' '.join(caption)
     	return caption

    def load_description(self):
    	text = open(self.file_name, 'r', encoding = 'utf-8').read()
    	mapping = dict()
    	for line in text.split("\n"):
    		token = line.split("\t")
    		if len(line) < 2: # remove short descriptions
    			continue
    		img_id = token[0].split('.')[0] # name of the image
    		img_des = token[1]			 # description of the image
    		if img_id not in mapping:
    			mapping[img_id] = list()
    		mapping[img_id].append(self.clean_description(img_des))
    	return mapping

    # load descriptions of training set in a dictionary. Name of the image will act as ey
    def load_clean_descriptions(self,des,dataset):
    	dataset_des = dict()
    	for key, des_list in des.items():
    		if key+'.jpg' in dataset:
    			if key not in dataset_des:
    				dataset_des[key] = list()
    			for line in des_list:
    				desc = 'startseq ' + line + ' endseq'
    				dataset_des[key].append(desc)
    	return dataset_des
    
    def preprocess_img(self,img_path):
    	img = load_img(img_path, target_size = (299, 299))
    	x = img_to_array(img)
    	x = np.expand_dims(x, axis = 0)
    	x = preprocess_input(x)
    	return x

    def Feature_extract(self,image_name,model):
    	image = self.preprocess_img(image_name)
    	vec = model.predict(image,verbose=0)
    	vec = np.reshape(vec, (vec.shape[1]))
    	return vec
    
    def load_train_images(self):
        
        descriptions = self.load_description()
        
        # Create a list of all image names in the directory
        images_name = glob.glob(self.images_path + '*.jpg')
        train_img = [i.split('\\')[-1] for i in images_name]
        
        train_descriptions = self.load_clean_descriptions(descriptions,train_img)
        
        base_model = Xception(weights = 'imagenet')
        model = Model(base_model.input, base_model.layers[-2].output)
        
        st = time.time();start = time.time();
        encoding_train = {}
        for i,img in enumerate(images_name):
            encoding_train[img[len(self.images_path):]] = self.Feature_extract(img,model)
            if i%200 == 0:
                print('Completed >>> %d   Time taken >>> %d sec'%((i/len(images_name))*100,time.time()-st))
                st = time.time()
        print('Completed >>> %d   Time taken >>> %d sec'%((i/len(images_name))*100,time.time()-start))
        return encoding_train,train_descriptions
    
    def caption_Train(self,train_descriptions):

        word_counts = {}
        max_length = 0
        for key, val in train_descriptions.items():
        	for caption in val:
        		cap = caption.split(' ')
        		for word in cap:word_counts[word] = word_counts.get(word, 0) + 1
        		if len(cap) > max_length:max_length = len(cap)
        
        # consider only words which occur atleast 10 times
        threshold = 10 # you can change this value according to your need
        vocab = [word for word in word_counts if word_counts[word] >= threshold]
        
        # word mapping to integers
        ixtoword = {};wordtoix = {};ix = 0
        for word in vocab:
        	wordtoix[word] = ix
        	ixtoword[ix] = word
        	ix += 1
        return max_length,vocab,ixtoword,wordtoix 

    def prepare_data(self,train_descriptions,encoding_train):
        
        max_length,vocab,ixtoword,wordtoix = self.caption_Train(train_descriptions)
        vocab_size = len(vocab)
        
        X_img, X_seq, y = list(), list(), list()
        for key, des_list in train_descriptions.items():
        	pic = encoding_train[key + '.jpg'] 
        	for cap in des_list:
        		seq = [wordtoix[word] for word in cap.split(' ') if word in wordtoix]
        		for i in range(1, len(seq)):
        			in_seq, out_seq = seq[:i], seq[i]
        			in_seq = pad_sequences([in_seq], maxlen = max_length)[0]
        			out_seq = to_categorical([out_seq], num_classes = vocab_size)[0]
        			# store
        			X_img.append(pic)
        			X_seq.append(in_seq)
        			y.append(out_seq)
        
        X_img = np.array(X_img)
        X_seq = np.array(X_seq)
        y = np.array(y)
        return X_img, X_seq, y, ixtoword, wordtoix 
    
    def Saving(self,X_img,X_seq,y,ixtoword,wordtoix):
        
        if os.path.exists('Features/'):
            pass
        else:
            os.makedirs('Features/')
            
        np.save('Features/X_img.npy',X_img)
        np.save('Features/X_seq.npy',X_seq)
        np.save('Features/y.npy',y)

        pickle.dump(ixtoword,open('Features/ixtoword.npy','wb'))
        pickle.dump(wordtoix,open('Features/wordtoix.npy','wb'))
    
    def load_data(self,save=False):
        encoding_train,train_descriptions = self.load_train_images()
        X_img, X_seq, y, ixtoword, wordtoix = self.prepare_data(train_descriptions,encoding_train)
        if save:
            self.Saving(X_img,X_seq,y,ixtoword,wordtoix)
        return X_img, X_seq, y, ixtoword, wordtoix 
    
    def load_saved_data(self,path='Features/'):
        
        if os.path.exists(path):
            pass
        else:
            raise Exception('Cannot identify Features directory in the path')
        
        X_img = np.load('Features/X_img.npy')
        X_seq = np.load('Features/X_seq.npy')
        y = np.load('Features/y.npy')

        ixtoword = pickle.load(open('Features/ixtoword.npy','rb'))
        wordtoix = pickle.load(open('Features/wordtoix.npy','rb'))
        
        return X_img, X_seq, y, ixtoword, wordtoix 


class Train_model:
    
    def __init__(self,):
        self.emb_dim = 100
        
    def get_embeding_matrix(self,wordtoix):
        with open('glove.6B.100d.txt','rb') as f:
            emb = f.readlines()
        emb_mat = {}
        for em in emb:
            dim = em.split()
            val = np.asarray([float(e) for e in dim[1:]])
            emb_mat[str(dim[0])] = val
        
        matrix = np.zeros((len(wordtoix),self.emb_dim))
        for i,w in enumerate(wordtoix):
            mat = emb_mat.get(w)
            if mat:
                matrix[i] = mat
        return matrix
        
        
    def get_model(self,img_shp,vocab_size,max_length,wordtoix):
        
        ip1 = Input(shape = img_shp)
        fe1 = Dropout(0.2)(ip1)
        fe2 = Dense(256, activation = 'relu')(fe1)
        
        ip2 = Input(shape = (max_length, ))
        se1 = Embedding(vocab_size, self.emb_dim,name='Embedding')(ip2)
        se2 = Dropout(0.2)(se1)
        se3 = LSTM(256)(se2)
        
        decoder1 = add([fe2, se3])
        decoder1 = Dropout(0.2)(decoder1)
        decoder2 = Dense(256, activation = 'relu')(decoder1)
        outputs = Dense(vocab_size, activation = 'softmax')(decoder2)
        model = Model(inputs = [ip1, ip2], outputs = outputs)
        
        wei = self.get_embeding_matrix(wordtoix)
        for i,lay in enumerate(model.layers):
            if lay.name == 'Embedding':
                model.layers[i].trainable = False
                break
        
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
        model.summary()
        return model
    
    def Training(self):
        
        flicker = Flicker_Dataset()
        X_img, X_seq, y, ixtoword, wordtoix  = flicker.load_data(save=True)
        del(flicker)
        
        vocab_size = len(y[0])
        max_length = len(X_seq[0])
        img_shp = X_img.shape[1:]
        
        model = self.get_model(img_shp,vocab_size,max_length,wordtoix)
        
        with tf.device("/CPU:0"): 
            model.fit([X_img,X_seq], y, epochs = 20, batch_size = 256,verbose=True)
        
        model.save('caption.h5')
        
    def retrain_model(self,path):
        root_path = os.getcwd()
        model = load_model(root_path+"\\"+path)

        flicker = Flicker_Dataset()
        X_img, X_seq, y, ixtoword, wordtoix  = flicker.load_saved_data()
    
        del(flicker)
        
        with tf.device("/CPU:0"): 
            model.fit([X_img,X_seq], y, epochs = 20, batch_size = 256,verbose=True)
        
        model.save(root_path+"\\"+path)

#%%

train = Train_model() 
train.Training()  

# train = Train_model() 
# train.retrain_model('caption.h5')

