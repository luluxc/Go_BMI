filename = '/home/appuser/venv/lib/python3.9/site-packages/keras_vggface/models.py'
text = open(filename).read()
open(filename, 'w+').write(text.replace('keras.engine.topology', 'tensorflow.keras.utils'))

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_probability as tfp
import keras.utils as image
from keras_vggface import VGGFace
from keras.models import Model
from keras_vggface.utils import preprocess_input
import numpy as np
import time
import io
import joblib
import pandas as pd
import av
import logging
import os
from twilio.rest import Client

logger = logging.getLogger(__name__)

os.environ["TWILIO_ACCOUNT_SID"] = "ACd01b2689b38f000027e44133cb446ba6"
os.environ["TWILIO_AUTH_TOKEN"] = "654a358bcbb4b8a54cb81f035a913c66"


def get_ice_servers():
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."  # noqa: E501
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    return token.ice_servers

#def pearson_corr(y_test, y_pred):
#  corr = tfp.stats.correlation(y_test, y_pred)
#  return corr

# def custom_object_scope(custom_objects):
#   return tf.keras.utils.CustomObjectScope(custom_objects)

# with custom_object_scope({'pearson_corr': pearson_corr}):
#   model = load_model('My_model_vgg16.h5')

#model = load_model('/content/gdrive/MyDrive/Colab Notebooks/My_BMI/My_model_vgg16.h5', compile=False)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# def predict_class(image, model):
#   img = image.copy()
#   img = cv2.resize(img, (224, 224))
#   img = np.array(img).astype(np.float32)
#   img = np.expand_dims(img, axis = 0)
#   img = preprocess_input(img, version=2)
#   prediction = model.predict(img)[0][0]
#   return prediction



# def process_img(file_image):
#   image = Image.open(file_image)
#   image = np.array(image)
#   gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#   faces = faceCascade.detectMultiScale(gray_image, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30))
#   if len(faces) == 0:
#     col2.write('No face detected! Please take it again.')
#   for (x, y, w, h) in faces:
#     # box bounding the face
#     cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
#     bmi = predict_class(image[y:y+h, x:x+w], model)
#     cv2.putText(image, f'BMI:{bmi}', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#   return 

# def predict_bmi(frame):
#     pred_bmi = []
#     faces = faceCascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),scaleFactor = 1.15,minNeighbors = 5,minSize = (30,30),)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
#         image = frame[y:y+h, x:x+w]
#         img = image.copy()
#         img = cv2.resize(img, (224, 224))
#         img = np.array(img).astype(np.float64)
#         features = get_fc6_feature(img)
#         preds = svr_model.predict(features)
#         pred_bmi.append(preds[0])
#         cv2.putText(frame, f'BMI: {preds}', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
#     return pred_bmi

def calculator(height, weight):
  return 730 * weight / height**2
  
def change_photo_state():
  st.session_state['photo'] = 'Done'


def load_svr():
    return joblib.load('svr_model.pkl')


def load_vggface():
    vggface = VGGFace(model='vgg16', include_top=True, input_shape=(224, 224, 3), pooling='avg')
    return Model(inputs=vggface.input, outputs=vggface.get_layer('fc6').output)

svr_model = load_svr()
vggface_model = load_vggface()

def get_fc6_feature(img):
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img, version=2) 
    fc6_feature = vggface_model.predict(img)
    return fc6_feature

def predict_bmi(frame):
    pred_bmi = []
    faces = faceCascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor = 1.15,minNeighbors = 5,minSize = (30,30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        image = frame[y:y+h, x:x+w]
        img = image.copy()
        img = cv2.resize(img, (224, 224))
        img = np.array(img).astype(np.float64)
        features = get_fc6_feature(img)
        preds = svr_model.predict(features)
        pred_bmi.append(preds[0])
        cv2.putText(frame, f'BMI: {preds - 4}', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return pred_bmi, frame

def prepare_download(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    image_bytes = buf.getvalue()
    return image_bytes

class VideoProcessor:
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.out_image = None
        self.pred_bmi = []

    def recv(self, frame):
        frm = frame.to_ndarray(format='bgr24')
        pred_bmi, frame_with_bmi = predict_bmi(frm)
        with self.frame_lock:
            self.out_image = frame_with_bmi
            self.pred_bmi = pred_bmi

        return av.VideoFrame.from_ndarray(frame_with_bmi, format='bgr24') 

def main():
  if 'photo' not in st.session_state:
    st.session_state['photo'] = 'Not done'

  st.set_page_config(layout="wide", page_icon='random', )
  st.markdown("""
  <style>
  .big-font {
      font-size:80px !important;
  }
  </style>
  """, unsafe_allow_html=True)

  st.markdown('<p class="big-font">BMI Prediction 📸</p>', unsafe_allow_html=True)
  bmi_img = Image.open('bmi.jpeg')
  st.image(bmi_img)
  #st.title('*BMI prediction 📸*')
  st.write('Body Mass Index(BMI) estimates the total body fat and assesses the risks for diseases related to increase body fat. A higher BMI may indicate higher risk of developing many diseases.')
  st.write('*Since we only have the access to your face feature, the estimated value is biased')
  
  webrtc_streamer(key="example",video_transformer_factory=VideoProcessor,rtc_configuration={'iceServers': get_ice_servers()},sendback_audio=False)
  
  col2, col3 = st.columns([2,1])
  upload_img = col3.file_uploader('Upload a photo 🖼', on_change=change_photo_state)
  file_image = col2.camera_input('Take a pic of you 😊', on_change=change_photo_state)         

  if st.session_state['photo'] == 'Done':
    process_bar3 = col3.progress(0, text='🏃‍♀️')
    process_bar2 = col2.progress(0, text='🏃')

    if file_image:
      for process in range(100):
        time.sleep(0.01)
        process_bar2.progress(process+1)
      col2.success('Taken the photo sucessfully!')
      file_image = np.array(Image.open(file_image))
      pred_camera = predict_bmi(file_image)
      ready_cam = Image.fromarray(file_image)
      col2.image(ready_cam, clamp=True)
      # Convert the PIL Image to bytes
      download_cam = prepare_download(ready_cam)
      col3.divider()
      col3.write('Download the predicted image if you want!')
      download_img = col3.download_button(
        label=':black[Download image]', 
        data=download_cam,
        file_name='BMI_image_camera.png',
        mime="image/png",
        use_container_width=True)
    elif upload_img:
      for process in range(100):
        time.sleep(0.01)
        process_bar3.progress(process+1)
      col3.success('Uploaded the photo sucessfully!')
      upload_img = np.array(Image.open(upload_img).convert('RGB'))
      pred_upload = predict_bmi(upload_img)
      ready_upload = Image.fromarray(upload_img)
      col2.image(ready_upload, clamp=True)
      # Convert the PIL Image to bytes
      download_img = prepare_download(ready_upload)
      col3.write('Download the predicted image if you want!')
      download_img = col3.download_button(
        label='Download image', 
        data=download_img,
        file_name='BMI_image_uploaded.png',
        mime="image/png")
  
  
  index = {'BMI':['16 ~ 18.5', '18.5 ~ 25', '25 ~ 30', '30 ~ 35', '35 ~ 40', '40~'],
           'WEIGHT STATUS':['Underweight', 'Normal', 'Overweight', 'Moderately obese', 'Severely obese', 'Very severely obese']}
  df = pd.DataFrame(data=index)
  hide_table_row_index = """<style>
                            thead tr th:first-child {display:none}
                            tbody th {display:none}
                            </style>"""
  col3.markdown(hide_table_row_index, unsafe_allow_html=True)
  col3.table(df)
  expander = col3.expander('BMI Index')
  expander.write('The table above shows the standard weight status categories based on BMI for people ages 20 and older. (Note: This is just the reference, please consult professionals for more health advices.)')
  
  
  col3.title('BMI calculator')
  cal = col3.container()
  with cal:
    feet = col3.number_input(label='Height(feet)')
    inch = col3.number_input(label='Height(inches)')
    weight = col3.number_input(label='Weight(pounds)')
    if col3.button('Calculate BMI'):
      if feet == 0.0:
        col3.write('Please fill in your heright(feet)')
      elif weight == 0.0:
        col3.write('Please fill in your weight(pounds)')
      else:
        height = feet * 12 + inch
        score = calculator(height, weight)
        col3.success(f'Your BMI value is: {score}')
      
if __name__=='__main__':
    main()
