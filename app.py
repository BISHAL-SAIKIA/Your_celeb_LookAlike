import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

detector=MTCNN()
st.title("Your Celeb Look Alike")

feature_list=pickle.load(open('embeddings.pkl','rb'))
filenames = pickle.load(open('filenames.pkl','rb'))
def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('Uploads',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),pooling='avg')

def extract_features(img_path, model,detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)
    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]

    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)
    face_array = face_array.astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()
    return result


def recommend(feature_list,features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index

uploaded_image = st.file_uploader('Choose and image')
if uploaded_image is not None:
    #save image in a directory uploads
    if save_uploaded_image(uploaded_image):
        #load the image
        #extract the features
        #recommend
        display_image = Image.open(uploaded_image)
        # st.image(display_image)
        #extract the features
        features= extract_features(os.path.join('Uploads',uploaded_image.name),model,detector)
        # st.text(features)
        # st.text(features.shape)
        #recommend
        index = recommend(feature_list,features)
        predicted_actor=" ".join((filenames[index].split('\\')[1].split('_')))
        #display
        # st.text(index)


        col1,col2 = st.beta_columns(2)

        with col1:
            st.header('Your uploaded image')
            st.image(display_image)
        with col2:
            st.header("Seems like "+predicted_actor)
            st.image(filenames[index],width=400)