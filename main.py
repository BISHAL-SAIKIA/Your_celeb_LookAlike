# import os
# import pickle
#
# actors=os.listdir('data')
#
# filenames=[]
# for actor in actors:
#     for file in os.listdir(os.path.join('data',actor)):
#         filenames.append(os.path.join('data',actor,file))
#
# # print(filenames)
# pickle.dump(filenames,open('filenames.pkl','wb'))

# WE CREATED A PICKLE FILE WHICH CONTAINS THE IMAGES OF THE ACTOS IN BINARY FORM.

from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm

filename = pickle.load(open('filenames.pkl' , 'rb'))

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),pooling='avg')
# print(model.summary())

def feature_extractor(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    image_array=image.img_to_array(img)
    expanded_img = np.expand_dims(image_array,axis=0) #will add one extra dimension
    preprocessed_input= preprocess_input(expanded_img)

    result= model.predict(preprocessed_input).flatten()

    return result

features=[]

for file in tqdm(filename):
    features.append(feature_extractor(file,model))

pickle.dump(features,open('embeddings.pkl','wb'))
