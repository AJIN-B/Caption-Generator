from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Model
import pickle;import os;import numpy as np
import argparse
import cv2

class Testing:
    
    def __init__(self,arg):
        self.model = load_model(arg.model_path)
        self.wordtoix = pickle.load(open('Features/wordtoix.npy','rb'))
        self.max_length = 34
        self.ixtoword = pickle.load(open('Features/ixtoword.npy','rb'))
        self.vocab_size = len(self.wordtoix)
        self.arg = arg
        self.generate_caption()
    
    def preprocess_img(self,img_path):
    	img = load_img(img_path, target_size = (299, 299))
    	x = img_to_array(img)
    	x = np.expand_dims(x, axis = 0)
    	x = preprocess_input(x)
    	return x

    def Feature_extract(self,image_name):
    	base_model = Xception(weights = 'imagenet')
    	model = Model(base_model.input, base_model.layers[-2].output)
    	image = self.preprocess_img(image_name)
    	vec = model.predict(image,verbose = self.arg.v)
    	return vec
    
    def Extract_sequence(self,img_path):
        img_feature = self.Feature_extract(img_path)
        start = 'startseq'
        for i in range(self.max_length):
            seq = [self.wordtoix[word] for word in start.split() if word in self.wordtoix]
            seq = pad_sequences([seq], maxlen = self.max_length)
            yhat = self.model.predict([img_feature, seq],verbose = self.arg.v)
            yhat = np.argmax(yhat)
            word = self.ixtoword[yhat]
            start += ' ' + word
            if word == 'endseq':break
        final = start.split()
        final = final[1:-1]
        final = ' '.join(final)
        return final 
    
    def display_image(self,i):
        img = cv2.imread(i)
        cv2.imshow('Input image',img)
        cv2.waitKey(1000)

    def generate_caption(self):
        
        if self.arg.img_path is list:
            for i in self.arg.img_path:
                if os.path.isfile(i):
                    sequence = self.Extract_sequence(i) 
                    self.display_image(i)
                    print('\nDescribing the image : ',sequence,'\n')
        else:
            sequence = self.Extract_sequence(self.arg.img_path) 
            self.display_image(self.arg.img_path)
            print('\nDescribing the image : ',sequence)

def Arguments():
    test_img = os.listdir('Test images/')
    rand_cho = np.random.choice(test_img)
    # Initialize the Parser
    parser = argparse.ArgumentParser()
    # Adding Arguments
    parser.add_argument('-mp','--model_path',default = 'caption.h5', type = str,
                        help ='Enter the path of the pretrained model')
    parser.add_argument('-t','--img_path',default = 'Test images/'+rand_cho,nargs = '*',
                        help ='Enter the path of image or images for testing') 
    parser.add_argument('-v','--v',default = True,help ='Verbose ie True or False')
      
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = Arguments()
    Testing(args)

