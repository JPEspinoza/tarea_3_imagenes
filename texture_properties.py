"""
This code processes the texture properties of every image in the folder "Image_set" 
and stores it into a CSV named texture_properties.csv
"""

from skimage.io import imread_collection
from skimage.feature import greycomatrix, greycoprops
import numpy as np
import pandas as pd
import cv2

images = imread_collection("Image_set/*.bmp")

# the features to read
features = ['contrast','correlation', 'dissimilarity','homogeneity','ASM','energy']

# here we store the properties of each image in each row
properties_matrix = []

# loop over every image
for image in images:

    grey =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # get comatrix in 4 angles
    comatrix = greycomatrix(
        image=grey, 
        distances=[1], 
        angles=[
            np.radians(0), 
            np.radians(45), 
            np.radians(90), 
            np.radians(135),
        ]
    )

    # make properties dictionary to have everything nicely indexed
    properties = {}

    # get features
    # the features are defined outside the for loop!
    for feature in features:
        property = greycoprops(comatrix, feature)

        # now we actually have a problem since we calculated 4 angles
        # we have four values for every property, equivalent to the four angles
        # we can actually use this, by adding all the angles together we should get
        # a rotationally invariant property of the texture!!

        properties[feature] = np.sum(property)
    
    # we append the dictionary to the matrix, each row representing an image
    properties_matrix.append(properties)

# transform our property matrix into a dataframe and store it
properties_df = pd.DataFrame(properties_matrix)
properties_df.to_csv("texture_properties.csv")