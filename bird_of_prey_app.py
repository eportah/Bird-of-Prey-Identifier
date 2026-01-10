import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.applications.mobilenet_v2 import preprocess_input

#establish path and class names
MODEL_PATH = 'bird_of_prey_identifier.keras'
CLASS_NAMES = [
    'Eagle', 'Falcon', 'Hawk', 'Owl', 'Vulture', 'Not a Bird Of Prey'
   ]

#cache command to load model once rather than each time an image is uploaded
@st.cache_resource
def load_trained_model():
  model = tf.keras.models.load_model(MODEL_PATH, compile=False)
  return model

#page layout
st.set_page_config(page_title="Bird Of Prey Identifier")
st.title("Bird of Prey Identifier")
st.write(
  "Upload an image of a bird to determine if it's a bird of prey or not"
  )

#load model 
model = load_trained_model()

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

  img_array = tf.keras.utils.img_to_array(img)
  img_batch = np.expand_dims(img_array, axis=0)

  img_preprocessed = preprocess_input(img_batch)

  prediction = model.predict(img_preprocessed)
  predicted_index = np.argmax(prediction[0])
  confidence = np.max(prediction[0]) * 100

  specific_bird = CLASS_NAMES[predicted_index]

  if specific_bird == 'notBirdOfPrey':
    result_spec_bird = "Not a bird of prey"
  else:
    result_spec_bird = f"Bird of Prey ({specific_bird.capitalize()})"

  return result_spec_bird, confidence

#file uploader
uploaded_file = st.file_uploader(
    "Upload a bird image...", type=["jpg", "jpeg", "png"]
    )

#classify uploaded image and yield prediction and confidence
"""
st.image to open file as an image
st.button to create Classify button
st.spinner for a loading message
st.success and st.info for results
"""
if uploaded_file:
  image = Image.open(uploaded_file)
  st.image(image, caption='Uploaded image', use_container_width=True)

  if st.button('Identify'):
    model = load_trained_model()
    label, score = predict(image, model)
    st.write(f"**Result:**{label} ({Score: .1f}%)")
