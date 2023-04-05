# Caption-Gereator
  In this project Flicker8k dataset was used to train the model for describing about the given image.Read images from dataset.
  Preprocessing the text data with punctucation removed and lowercase the string.
  After for each sequence was encode and fed into the train model.
  For extracting the image feature Xception is used and for the sequence to train LSTM layer has been used and these two combine we get the model.
  After trained with the model the model is well perofmed with the test images to describing about the given image.

For Training first download the fliker8k dataset.

After arrange the files like this

Caption Generator

\t-> Feature
   
   -> Flicker_Text 
        # inside this folder put all the text files
        
   -> Flicker8k_Dataset 
        # inside this folder put all images
        
   -> Test images 
        # testing images
        
   -> glove.6B.100d # glove embedding vector 
        # you want to download this file from stanford site glove embedding

After all changes done run the Training.py file

For testing 
you can run with just : python testing.py 
if you want to give any images :  python testing.py  --img_path #path for image

arguments are 

  --model_path # Enter the path of the pretrained model
  
  --img_path # 'Enter the path of image or images for testing
  
  --v # Verbose ie True or False

For Help in the testing file

python testing.py --help
