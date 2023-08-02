# Caption-Gereator

#### The aim of the project is used to describing about the given image.

#### In this project Flicker8k dataset used to train the LSTM model.

- Read every training images from the dataset.
- Preprocessing the text data like removing punctucation and lowercase the string.
- After for each sequence was encode and fed into the train model.
- For extracting the image feature **Xception Network** is used.
- For the sequence LSTM has been used.
- For Embedding layer the golve vectors has been fixed as weights and trainable set as False.
- After the model is trained with the training dataset
- After the trained model is the tested with the testing images to describing about the given image.

#### For download the fliker8k dataset [Download](https://www.kaggle.com/datasets/adityajn105/flickr8k) 

#### For Download the glove.6B.100d from the stanford glove embedding vector [Download](https://nlp.stanford.edu/projects/glove/) you want to 

#### Folder Arrangement

+-- Parent folder
+-- Features 
|   +-- It contains the feature that are saved while training  
+-- Flicker_Text
|   +-- It contains the text data in the .txt files
+-- Flicker8k_Dataset
|   +-- It contains the images
+-- Test images
|   +-- It contains some samples to test the model 
+-- glove.6B.100d # glove embedding vector
+-- caption.h5 # Trained model
+-- Training.py # training file
+-- Training_generator.py # training file using generator function
+-- testing.py # to testing the model


.
+-- _config.yml
+-- _drafts
|   +-- begin-with-the-crazy-ideas.textile
|   +-- on-simplicity-in-technology.markdown
+-- _includes
|   +-- footer.html
|   +-- header.html
+-- _layouts
|   +-- default.html
|   +-- post.html
+-- _posts
|   +-- 2007-10-29-why-every-programmer-should-play-nethack.textile
|   +-- 2009-04-26-barcamp-boston-4-roundup.textile
+-- _data
|   +-- members.yml
+-- _site
+-- index.html

#### After all have been done run the Training.py file using

`python Training.py`

or 

`python Training_generator.py`


### For testing 

`python testing.py `

#### Try with your own image

`python testing.py --img_path "\Test images\1000268201_693b08cb0e.jpg"` 

### Arguments for testing.py

- --model_path # Enter the path of the pretrained model
  
- --img_path # Enter the path of image or images for testing
  
- --v # Verbose ie True or False

#### For Help in the testing file

`python testing.py --help`
