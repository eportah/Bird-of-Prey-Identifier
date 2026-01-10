import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.applications.mobilenet_v2 import preprocess_input

#establish path and class names
MODEL_PATH = 'bird_of_prey_identifier.keras'
CLASS_NAMES = ['Bird of Prey', 'Not a Bird of Prey']

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

st.title("Bird of Prey Identifier")
uploaded_file = st.file_uploader("Upload an image of a bird", type=["png", "jpg", "jpeg"])

#image
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Identify"):
        with st.spinner("Analyzing..."):
      
            model = load_model()
          
            img = image.resize((224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_input(img_batch)
            
            prediction = model.predict(img_preprocessed)
            index = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            label = CLASS_NAMES[index]
            
            if label == 'Not a Bird of Prey':
                st.error(f"Result: {label} ({confidence:.1f}%)")
            else:
                st.success(f"Result: {label} ({confidence:.1f}%)")
