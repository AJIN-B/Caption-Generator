# Caption-Gereator (Generating Image Describtions)

###Constructed a sophisticated image caption generator using deep learning techniques to automatically describe images. 
This project harnessed the potential of both natural language processing and computer vision to generate coherent and 
contextually relevant captions for images.

### Project Overview 

This project is focused on leveraging computer vision and natural 
language processing (NLP) techniques to create a system capable of providing meaningful 
and contextually relevant descriptions for images. This technology can be applied in 
various domains, including accessibility, content tagging, search engine optimization, and more.
An Image Description project represents an exciting intersection of computer vision and 
natural language processing, aiming to bridge the gap between visual and textual information,
with the potential to benefit a broad range of users and industries.


**Dataset Used:** Employed the Flickr8k dataset as the primary source of data for this project, which includes a diverse range of images and corresponding descriptions.
In this dataset consisting of 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events .[Download](https://www.kaggle.com/datasets/adityajn105/flickr8k)

**Image Processing:**  The system processes input images using computer vision algorithms from the dataset.

**Describtions Processing:**  The text Preprocessing contains tokenization ,removal of punctucation ,lowercase and unique value to every words.

**Feature Extraction:**  Extract relevant features and attributes from the visual content using the **Xception Network**.This may include recognizing objects, colors, shapes, sizes, and spatial relationships.

**NLP Model:**  Implement a natural language processing model, such as a neural network, that takes 
the extracted image features as input and generates descriptive text as output. The model was trained on image-text pairs.

**Model Constrution:**  For extract image feature **Xception** has been used.
Then it has been fed into the **Dense** layer.The text vectors has been given into the **Embedding** layer for extract the textual context of the describtions.
The **Embedding layer weights** has been replaced as **glove vectors embedding**.
Then extracted image and text features has been fed into deep neural network **Long Short Term Memory (LSTM)** and a **fully connected Dense** layer with the activation **softmax function**.

**Training:**  Then the model has been trained with 80% of train data.Total no of epochs is 100.**The model has been trained in GPU RTX 3050 using cuda libraries.**

**Performance Evaluation:** Evaluated the model's performance with testing data, demonstrating its competence in accurately generating descriptive captions for images.

**Technologies and Tools:** Deep Learning, Natural Language Processing, Computer Vision, TensorFlow, OpenCV, NLTK, LSTM


#### For download the glove.6B.100d from the stanford glove embedding vector [Link](https://nlp.stanford.edu/projects/glove/)  

#### Folder Arrangement

```
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
```


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

**Challenges:**
- Ensuring the generated descriptions are accurate and coherent.
- Handling images with complex scenes and a wide range of objects.
- Addressing the trade-off between verbosity and conciseness in descriptions.

**Benefits:**
- Automation of image description for a variety of applications.


## Installation 

### Basic Requirements:

- __[Python 3.10](https://docs.python.org/3/)__
- __[Tensorflow](https://pypi.org/project/tensorflow/)__ 
- __[Opencv-python](https://pypi.org/project/opencv-python/)__ 
- __[Pandas](https://pandas.pydata.org/docs/)__
- __[Numpy](https://numpy.org/doc/)__ 

### To install the basic Requirements

`pip install - r requirements.txt`

