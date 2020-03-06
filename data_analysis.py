# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
"""
build an algorithm to detect a visual signal for pneumonia in medical images.
Specifically, your algorithm needs to automatically locate lung opacities on chest 
radiographs.

Pneumonia usually manifests as an area or areas of increased opacity [3] on CXR.
However, the diagnosis of pneumonia on CXR is complicated because of a number of other
conditions in the lungs such as fluid overload (pulmonary edema), bleeding, volume loss 
(atelectasis or collapse), lung cancer, or post-radiation or surgical changes

Webpage of the challenge: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview/description

"""
###############################################################################
####################  DATA ANALYSIS AND OBSERVATIONS  #########################
###############################################################################

import matplotlib.pyplot as plt
import glob, pylab, pandas as pd
import pydicom, numpy as np
import seaborn as sns




#######          CLASS INFOS         #########

df_class_info = pd.read_csv('/Users/Arthur/Desktop/ULB/2019-2020/Projet_Imagerie/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv')

print("\n Class information of the first 10 patients \n")
print(df_class_info.head(10))
print("\n Class info distribution \n")

f, ax = plt.subplots(1,1, figsize=(6,4))
total = float(len(df_class_info))
sns.countplot(x=df_class_info['class'] ,order = df_class_info['class'].value_counts().index)#, palette='Set3')
#sns.countplot parameters: x= define the x axis; palette is for the colors ; order is a list of strings, order to plot the categorical lebels in
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(100*height/total),
            ha="center") 
plt.show()

#######           TRAIN LABELS         #########


df_train_labels = pd.read_csv('/Users/Arthur/Desktop/ULB/2019-2020/Projet_Imagerie/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')
print("\n First 5 patients: train labels \n")
print(df_train_labels.head(5))# Target is a binary variable: 0 if normal and 1 if pneumonia

#print(df_train_labels.iloc[4])
#print(type(df_train_labels))

def parse_data(df_train_labels):
    """
    Method to read a CSV file (Pandas dataframe) and parse the 
    data into the following nested dictionary:

      parsed = {
        
        'patientId-00': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        },
        'patientId-01': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        }, ...

      }

    """
    # --- Define lambda to extract coords in list [y, x, height, width]
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]#Return a list
    #Lambda functions is a short function in one line: lambda arguments : expression

    parsed = {} #dictionnary
    for n, row in df_train_labels.iterrows():#Iterrate over dataframe rows
        # --- Initialize patient entry into parsed 
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'file': '/Users/Arthur/Desktop/ULB/2019-2020/Projet_Imagerie/rsna-pneumonia-detection-challenge/stage_2_train_images/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed

parsed = parse_data(df_train_labels)

print("\n Display of the three first patients after conversion of the datas into a dictionary form \n")

for x in list(parsed)[0:3]:
    print ("key {}, value {} \n".format(x,  parsed[x]))
    

#######           TRAIN IMAGES         #########


""" Medical images are stored in a special format known as DICOM files (*.dcm). 
They contain a combination of header metadata as well as underlying raw image arrays 
for pixel data 
"""

patientId = df_train_labels['patientId'][4]
dcm_file = '/Users/Arthur/Desktop/ULB/2019-2020/Projet_Imagerie/rsna-pneumonia-detection-challenge/stage_2_train_images/%s.dcm' % patientId
dcm_data = pydicom.read_file(dcm_file) # method allows to read the data of dcm_file

print("\n Dicom metadatas of patient 4 radiography \n")
print(dcm_data)  # We can see that the images have already been rescaled and resized.

im = dcm_data.pixel_array #Dataset.pixel_array returns a numpy.ndarray containing the pixel data
pylab.imshow(im, cmap=pylab.cm.gist_gray)
pylab.axis('off')

# Visualizing boxes in on images

def draw(data):
    
    """
    Method to draw single patient with bounding box(es) if present 

    """
    # --- Open DICOM file
    d = pydicom.read_file(data['file'])
    im = d.pixel_array #Dataset.pixel_array returns a numpy.ndarray containing the pixel data
    # shape= (1024,1024)
    # --- Convert from single-channel grayscale to 3-channel RGB
    im = np.stack([im] * 3, axis=2) #np.stack(): Join a sequence of arrays along a new axis. We have 3times array im with dimension=3 as dimension1 is referenced by 0
    #shape=(1024,1024,3)
    #print(im.shape)#Passage a 3D array with last dimension = rgb

    # --- Add boxes with random color if present
    for box in data['boxes']:#box=liste de 4valeurs parmis les values du dico
        rgb = np.floor(np.random.rand(3)*256).astype('int')# 3=size of array; juste histoire d'avoir 3 couleurs pour R, G and B
        #rgb = np.array de trois valeurs
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)

    pylab.imshow(im, cmap=pylab.cm.gist_gray)
#    pylab.axis('off')

def overlay_box(im, box, rgb, stroke=1): # What represent stroke?
    """
    Method to overlay single box on image

    """
    # --- Convert coordinates to integers
    box = [int(b) for b in box] #ok
    #print(box)
    
    # --- Extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width
    
    #print(type(stroke))
    im[y1:y1 +stroke , x1:x2] = rgb 
    im[y2:y2 +stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im
print("\n Boxing of lung opacities of patient 4 \n")
draw(parsed[patientId])
























