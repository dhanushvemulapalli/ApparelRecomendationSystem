from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import urllib.request

# from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import math
import time
import re
import os
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances
from matplotlib import gridspec
from scipy.sparse import hstack
import plotly
import plotly.figure_factory as ff
from plotly.graph_objs import Scatter, Layout

import tensorflow as tf
import numpy as np
# from keras.preprocessing.image import ImageDataGenerator
# import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import requests
import pandas as pd
import pickle

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg



data = pd.read_pickle('16k_apperal_data_preprocessed')
print(data.columns)
# def display_img(url,ax,fig):
#     # we get the url of the apparel and download it
#     response = requests.get(url)
#     img = Image.open(BytesIO(response.content))
#     # we will display it in notebook 
#     plt.imshow(img)
import PIL.Image

# def display_img(url):
#     # Open the URL and convert to an image
#     # response = requests.get(url)
#     # image_data = BytesIO(response.content)
#     # image = Image.open(image_data)
#     # image.show()
#     if url == None:
#         print("No image found")
#     else:
#         urllib.request.urlretrieve(url, "image.jpg")
#         img = Image.open("image.jpg")
#         img.show()

    # response = requests.get(url)
    # img = Image.open(BytesIO(response.content))
    # # we will display it in notebook
    # plt.imshow(img)
"""
    Convert image to numpy array
    img_np = np.array(img)

    # Display the image
    plt.imshow(img_np)
    plt.axis('off')  # Hide axes
    plt.show()
    """

def display_img(url):
    # Send a GET request to the URL to fetch the image data
    response = requests.get(url)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open the image from the response content using PIL
        img = Image.open(BytesIO(response.content))
        
        # Display the image
        img.show()
    else:
        print("Failed to fetch image from URL")


from IPython.display import display, Image, SVG, Math, YouTubeVideo
bottleneck_features_train = np.load('16k_data_cnn_features.npy')
asins = np.load('16k_data_cnn_feature_asins.npy')

data = pd.read_pickle('16k_apperal_data_preprocessed')
df_asins = list(data['asin'])
asins = list(asins)

# tfidf_title_vectorizer = TfidfVectorizer(min_df = 0)
# tfidf_title_features = tfidf_title_vectorizer.fit_transform(data['title'])
# tfidf_title_features = tfidf_title_features.toarray()

# tfidf_desc_vectorizer = TfidfVectorizer(min_df = 0)
# tfidf_desc_features = tfidf_desc_vectorizer.fit_transform(data['description'])
# tfidf_desc_features = tfidf_desc_features.toarray()
data['brand'] = data['brand'].astype(str)

title_vectorizer = CountVectorizer()
title_features   = title_vectorizer.fit_transform(data['title'])

tfidf_desc_vectorizer = CountVectorizer()
tfidf_desc_features = tfidf_desc_vectorizer.fit_transform(data['brand'])

def Neighbour_of_Product(doc_id, num_results, w1, w2, w3):

    doc_id = asins.index(df_asins[doc_id])
    image_dist = pairwise_distances(bottleneck_features_train, bottleneck_features_train[doc_id].reshape(1,-1))
    title_dist = pairwise_distances(title_features,title_features[doc_id])
    desc_dist = pairwise_distances(tfidf_desc_features, tfidf_desc_features[doc_id])
    pairwise_dist = (w1 * image_dist + w2 * title_dist + w3 * desc_dist)/float(w1 + w2 + w3)

    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]
    print('='*60)
    print('Recommendation for the product of')
    print('Product Title: ', data['title'].loc[data['asin']==asins[doc_id]].values[0])
    print('Product Image:', data['medium_image_url'].loc[data['asin']==asins[doc_id]].values[0])
    print('='*60)
    print('='*135)

    for i in range(len(indices)):
        rows = data[['medium_image_url','title']].loc[data['asin']==asins[indices[i]]]
        for indx, row in rows.iterrows():
            print('Product Title: ', row['title'])
            print('Euclidean Distance from input image:', pdists[i])
            print('Amazon Url: www.amzon.com/dp/'+ asins[indices[i]])
            print('image url:', row['medium_image_url'])
            # display_img(row['medium_image_url'])
            # urllib.request.urlretrieve(row['medium_image_url'], "image.jpg")
            # img = Image.open("image.jpg")
            # img.show()
            print('='*135)

Neighbour_of_Product(12566, 10,  5 , 7 , 5)