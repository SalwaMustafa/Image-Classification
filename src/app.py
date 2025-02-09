import streamlit as st
import numpy as np
from PIL import Image
from model import generate_ans

st.write("""
         # Natural Scenes Classification
         """
         )
file = st.file_uploader("Please upload a Natural Scenes image" , type = ["jpg" , "png" , "jpeg"])

if file is None :
    st.text("---")
else:

    image = Image.open(file)
    image1 = np.array(image)
    st.image(image1 ,caption="Uploaded Image", use_container_width = True)
    predicted_label = generate_ans(file)
    st.success(f"This image is most likely a : {predicted_label}")