from skimage.io import imread_collection
from math import copysign, log10
from skimage.feature import greycomatrix, greycoprops
from multiprocessing import Pool
import numpy as np
import pandas as pd
import cv2

# the features we are going to extract
features = ['contrast','correlation', 'dissimilarity','homogeneity','ASM','energy']

def mrs(r, s, I, J):
    i=I**r
    j=J**s
    return np.sum(i*j)

def  m_central(r, s, I, J):
    m00 = mrs(0,0,I,J)
    m10 = mrs(1,0,I,J)
    m01 = mrs(0,1,I,J)

    ci = m10/m00
    cj = m01/m00
    i = (I-ci)**r
    j = (J-cj)**s

    return sum(i*j)
    
def eta(r, s, I, J):
    t = (r+s)/2 + 1
    a =  m_central(r,s,I,J)
    b =  m_central(0,0,I,J)

    return a/(b**t)

def hu(I, J):
    H= np.zeros(7)
    eta11 = eta(1,1,I,J)
    eta12 = eta(1,2,I,J)
    eta20 = eta(2,0,I,J)
    eta21 = eta(2,1,I,J)
    eta02 = eta(0,2,I,J)
    eta03 = eta(0,3,I,J)
    eta30 = eta(3,0,I,J)

    H[0] = eta20+eta02
    H[1] = (eta20-eta02)**2 + 4*eta11**2
    H[2] = (eta30-3*eta12)**2+(3*eta21-eta03)**2
    H[3] = (eta30+eta12)**2+(eta21+eta03)**2
    H[4] =(eta30-3*eta12)*(eta30+eta12)*( (eta30+eta12)**2-3*(eta21+eta03)**2)+(3*eta21-eta03)*(eta21+eta03)*(3* (eta30+eta12)**2- (eta21+eta03)**2 )
    H[5] = (eta20-eta02)*((eta30+eta12)**2-(eta21+eta03)**2+ 4*eta11*(eta30+eta12)*(eta21+eta03))
    H[6] = (3*eta21-eta03)*(eta30+eta12)*((eta30+eta12)**2-3*(eta21+eta03)**2)+(eta30-3*eta12)*(eta21+eta03)* (3*(eta30+eta12)**2-(eta21+eta03)**2)
    return H

# extract everything we want from a single image
def extract(image):
    # a dict to store all properties with their indexes
    properties = {}

    # get grey and binary version, we are going to use them later
    grey =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grey, 125,255,cv2.THRESH_BINARY_INV)

    # get the hu moments
    cij = np.argwhere(binary==255)
    huMoments = hu(cij[:,0], cij[:,1])
    for i in range(0,7):
        properties[f"hu{i}"] = -1* copysign(1.0, huMoments[i]) * log10(np.abs(huMoments[i]))

    # get comatrix in 4 angles for that extra slowness
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

    # get features with the coocurrence matrix
    # the features are defined outside the for loop!
    for feature in features:
        property = greycoprops(comatrix, feature)

        # now we actually have a problem since we calculated 4 angles
        # we have four values for every property, equivalent to the four angles
        # we can actually use this, by adding all the angles together we should get
        # a rotationally invariant property of the texture!!

        properties[feature] = np.sum(property)
    
    return properties

#read all images
images = imread_collection("Image_set/*.bmp")

# we are going to store all properties of every image here,
# one row per image
# this will be later turned into a dataframe
properties_matrix = []

#serial version
#for image in images:
    #properties = extract(image)
    #properties_matrix.append(properties)

#multiprocess version
with Pool(processes=8) as pool:
    properties_matrix = pool.map(extract, images)

properties_df = pd.DataFrame(properties_matrix)
properties_df.to_csv("properties.csv", index=False)