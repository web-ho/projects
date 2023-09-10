import streamlit as st
from PIL import Image
import numpy as np
import json
import os

from ultralytics import YOLO
from utils import display_img

# load from the path
model_path = "model\\yolo8m_1080_25e.pt"
model = YOLO(model_path)

class_labels = {
    'coca-cola' : 0,
    'fanta' : 1,
    'sprite' : 2
}
class_names = class_labels.keys()


#st.set_page_config(
 #   page_title="Detect Soda Bottles",
  #  page_icon=":bottle:",
   # #layout="wide",
    #initial_sidebar_state="expanded",)

# display page text
image_path = os.path.abspath("media/01_20220314_211527_bmp_jpg.rf.77420494628c23aeffd80dc00ba9128b.jpg")
image = Image.open(image_path)
st.title('Detect And Count Bottles')
Info = """
    Simple demo app to showcase the capability of a model trained on images similar to the one displayed.
    Mean Absolute Error of the model is 0.08.
    """
st.write(Info)
st.image(image, caption='Training image sample')

st.markdown(
    """
    <h4 style="text-align: left; color: black;">
        <span style="font-size: small;"></span>
        You can select one image from the sibebar for prediction. 
        You can also upload any image from your system to detect and count bottles in an image.
        Once the image is selected click "Predict" and you will have the predictions(image + count).
        <span style="font-size: small;"></span>
    </h4>
    Make sure to unselect the image before using either of the options

    """,
    unsafe_allow_html=True
)

# make a dict to allow users to select image
image_dict = {
    "Image 1": os.path.abspath("media/01_20220316_194709_bmp_jpg.rf.6c482a0ec06071e2d6f42550a57c0f3f.jpg"),
    "Image 2": os.path.abspath("media/01_20220316_195551_bmp_jpg.rf.6268d444e93b7a1b3d28249a47071412.jpg"),
    "Image 3": os.path.abspath("media/01_20220317_125503_bmp_jpg.rf.0b23073522df8d72b131f7b3919b9731.jpg"),
    "Image 4": os.path.abspath("media/01_20220317_130344_bmp_jpg.rf.0bfab87dc2bd9cbc3966db4ddd2146ef.jpg"),
    "Image 5": os.path.abspath("media/01_20220317_142317_bmp_jpg.rf.d1c196cb7c0c17ef6ac50d2002449133.jpg"),
    "Image 6": os.path.abspath("media/01_20220317_142758_bmp_jpg.rf.44012a9055cc9dba376145df536c4ca6.jpg"),
    "Image 7": os.path.abspath("media/01_20220317_145211_bmp_jpg.rf.e711e8d61eed9f615d4e9332358dc88f.jpg"),
    "Image 8": os.path.abspath("media/01_20220317_145331_bmp_jpg.rf.29b9be60a756d846420c49f16e7d1367.jpg"),
    }

select_image = st.sidebar.selectbox(
    "Select an image", [""]+list(image_dict.keys()))


st.sidebar.write("### Three soft-drinks used are:")
for name in class_names:
    st.sidebar.write(f"-{name}")

if select_image:
    image_path = image_dict[select_image]
    image = Image.open(image_path)
    st.image(image, caption=select_image)  

# allow users to upload their own prefered image
img_file = st.file_uploader("Choose any image", type=["jpg", "png"])
if img_file:
    img = Image.open(img_file)
    st.image(img, caption="Uploaded image")   

# Make prediction on selected or uploaded image
if select_image or img_file:
    if st.button("Predict"):
        if select_image:
            result = model.predict(image_path, save=False, conf=0.1, iou=0.1, imgsz=1080)
        else:
            image_path = os.path.join("media", img_file.name)
            img.save(image_path)
            result = model.predict(image_path, save=False, conf=0.1, iou=0.1, imgsz=1080)
        fig, ax, c, f, s = display_img(image_path, results=result)
        st.pyplot(fig)  # Display the figure within Streamlit
        st.success("Prediction Done!")
        st.success(f"'Coca-Cola Bottles': {c},  'Fanta Bottles': {f},  'Sprite Bottles': {s}")
        #st.snow()



