import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input

#set up page
st.set_page_config(page_title="Bird Of Prey Identifier")
st.title("Bird of Prey Identifier")
st.write(
  "Upload image of a bird for the model to predict")

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
def predict_bird_of_prey(img_to_predict, model):
  img = img_to_predict.convert('RGB')
  img = img.resize((224, 224))
  img_array = image.img_to_array(img)

  img_batch = np.expand_dims(img_array, axis=0)

  img_preprocessed = preprocess_input(img_batch)

  prediction = model.predict(img_preprocessed)
  predicted_index = np.argmax(prediction[0])
  specific_bird = CLASS_NAMES[predicted_index]
  confidence = np.max(prediction[0]) * 100

  if specific_bird == 'notBirdOfPrey':
    result_spec_bird = "Not a bird of prey"
  else:
    result_spec_bird = f"Bird of Prey ({specific_bird.capitalize()})"

  return result_spec_bird, confidence

#load model and create file uploader
model = load_trained_model()
uploaded_file = st.file_uploader(
    "Upload an image...", type=["jpg", "jpeg", "png"]
    )

#classify uploaded image and yield prediction and confidence
"""
st.image to open file as an image
st.button to create Classify button
st.spinner for a loading message
st.success and st.info for results
"""
if uploaded_file is not None:
  pil_image = Image.open(uploaded_file)
  st.image(pil_image, caption='Uploaded image', use_column_width=True)
  if st.button('Classify'):
    with st.spinner('Classifying...'):
      predicted_class, confidence = predict_bird_of_prey(pil_image, model)
      st.success(f"Prediction: **{predicted_class}**")
      st.info(f"Confidence: **{confidence:.2f}%**")
