import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input

#set up page
st.set_page_config(pageTitle="Bird Of Prey Identifier")
st.title("Bird of Prey Identifier")
st.write("Upload an image of a bird for the model to predict if it's a bird of prey or not.")

#establish path
MODEL_PATH = 'birdOfPreyIdentifier.keras'

#cache command to load model once rather than each time an image is uploaded
@st.cache.resource
def loadTrainedModel():
  model = tf.keras.models.load_model(MODEL_PATH)
  return model

#define class names
CLASS_NAMES = ['eagle', 'falcon', 'hawk', 'owl', 'vulture', 'NotBirdsOfPrey']

#preprocess image and return predicted class and confidence score
"""
convert image to numpy array
add extra dimension for batch
preprocess image for model
get model's prediction and turn into human-readable result
get specific bird type
"""
def predictBirdOfPrey(img_to_predict, model):
  img = img_to_predict.convert('RGB')
  img = img.resize((224, 224))
  img_array = image.img_to_array(img)

  img_batch = np.expand_dims(img_array, axis=0)

  img_preprocessed = preprocess_input(img_batch)

  prediction = model.predict(img_preprocessed)
  predictedIndex = np.argmax(prediction[0])
  specificBird = CLASS_NAMES[predictedIndex]
  confidence = np.max(prediction[0]) * 100

  if specificBird == 'notBirdOfPrey':
    resultSpecBird = "Not a bird of prey"
  else:
    resultSpecBird = f"Bird of Prey ({specificBird.capitalize()})"

  return resultSpecBird, confidence

#load model and create file uploader
model = loadTrainedModel()
uploadedFile = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

#classify uploaded image and yield prediction and confidence
"""
st.image to open file as an image
st.button to create Classify button
st.spinner for a loading message
st.success and st.info for results
"""
if uploadedFile is not None:
  pil_image = Image.open(uploadedFile)
  st.image(pil_image, caption='Uploaded image', use_column_width=True)
  if st.button('Classify'):
    with st.spinner('Classifying...'):
      predictedClass, confidence = predictBirdOfPrey(pil_image, model)
      st.success(f"Prediction: **{predictedClass}**")
      st.info(f"Confidence: **{confidence:.2f}%**")
