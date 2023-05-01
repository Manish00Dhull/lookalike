# from keras_vggface.utils import preprocess_input
# from keras_vggface.vggface import VGGFace
# import numpy as np
# import pickle
# from sklearn.metrics.pairwise import cosine_similarity
# import cv2
# from mtcnn import MTCNN
# from PIL import Image
# import matplotlib.pyplot as plt
#
# feature_list = np.array(pickle.load(open('embeddding.pkl','rb')))
# filenames = pickle.load(open('filenames.pkl','rb'))
#
# model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
#
# detector = MTCNN()
#
# sample1_img = plt.imread('s6.jpg')
# results = detector.detect_faces(sample1_img)
#
# x,y,width,height = results[0]['box']
#
# face = sample1_img[y:y+height,x:x+width]
#
# cv2.imshow('output',face)
#print(results)

import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image

feature_list = np.array(pickle.load(open('embeddding.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))


model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

detector = MTCNN()

sample1_img = cv2.imread('s6.jpg')
results = detector.detect_faces(sample1_img)

x,y,width,height = results[0]['box']

face = sample1_img[y:y+height,x:x+width]

cv2.imshow('output',face)

cv2.waitkey(0)