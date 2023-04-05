import os
import numpy as np
from tensorflow.keras.layers import  Dense, LSTM, Dropout, Embedding, Input
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import add
import pickle


def load_saved_data(path='Features/'):
    
    if os.path.exists(path):
        pass
    else:
        raise Exception('Cannot identify Features directory in the path')
    
    X_img = np.load(path+'X_img.npy')
    X_seq = np.load(path+'X_seq.npy')
    y = np.load(path+'y.npy')

    ixtoword = pickle.load(open('Features/ixtoword.npy','rb'))
    wordtoix = pickle.load(open('Features/wordtoix.npy','rb'))
    
    return X_img, X_seq, y, ixtoword, wordtoix 


class Train_model:
    
    def __init__(self,):
        self.emb_dim = 100
        
    def get_embeding_matrix(self,wordtoix):
        with open('E:/work/Caption Generator/glove.6B.100d.txt','rb') as f:
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


def generator(batch_size,X_img, X_seq, y):
    tot_batch = len(X_img)//batch_size
    for i in range(tot_batch):
        st = i*batch_size
        ed = (i+1)*batch_size
        yield [X_img[st:ed] ,X_seq[st:ed] ] ,y[st:ed]
    if ed < len(X_img):
        yield [X_img[ed:] ,X_seq[ed:] ] ,y[ed:]


if __name__ == "__main__":
    X_img, X_seq, y, ixtoword, wordtoix  = load_saved_data()
    
    vocab_size = len(y[0])
    max_length = len(X_seq[0])
    img_shp = X_img.shape[1:]
    
    train = Train_model() 
    model = train.get_model(img_shp,vocab_size,max_length,wordtoix)
    
    batch_size = 256
    for i in range(20):
        g = generator(batch_size,X_img, X_seq, y)
        model.fit(g, epochs = 1, verbose=True)
    
        


