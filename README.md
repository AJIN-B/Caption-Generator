# Caption-Gereator

#### The aim of the project is used to describing about the given image.

###### In this project Flicker8k dataset used to train the LSTM model.

- Read every training images from the dataset.
- Preprocessing the text data like removing punctucation and lowercase the string.
- After for each sequence was encode and fed into the train model.
- For extracting the image feature **Xception** is used and for the sequence to train LSTM layer has been used and these two combine we get the model.
- After trained with the model the model is well perofmed with the test images to describing about the given image.

##### For training first download the fliker8k dataset [Download](https://www.kaggle.com/datasets/adityajn105/flickr8k) 

##### Download the glove.6B.100d from the stanford glove embedding vector [Download](https://nlp.stanford.edu/projects/glove/) you want to 


#### Folder Arrangement
+-- Parent folder
    +-- Features 
        +-- It contains the feature that are saved while training
    +-- Flicker_Text
        +-- It contains the text data in the .txt files
    +-- Flicker8k_Dataset
        +-- It contains the images
    +-- Test images
        +-- It contains some samples to test the model 
    +-- glove.6B.100d # glove embedding vector
    +-- caption.h5 # Trained model
    +-- Training.py # training file
    +-- Training_generator.py # training file using generator function
    +-- testing.py # to testing the model



After arrange the files like this

Caption Generator

> Feature
  
> Flicker_Text # inside this folder put all the text files
        
> Flicker8k_Dataset # inside this folder put all images
        
> Test images # testing images
        
> glove.6B.100d # glove embedding vector [link](https://nlp.stanford.edu/projects/glove/) you want to download this file from stanford site glove embedding

#### After all changes done run the Training.py file

For testing 
you can run with just : python testing.py 
if you want to give any images :  python testing.py  --img_path #path for image

arguments are 

  --model_path # Enter the path of the pretrained model
  
  --img_path # 'Enter the path of image or images for testing
  
  --v # Verbose ie True or False

For Help in the testing file

python testing.py --help
