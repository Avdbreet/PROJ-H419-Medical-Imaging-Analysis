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


import glob, pylab, pandas as pd
import pydicom, numpy as np

# TRAIN LABELS

df = pd.read_csv('/Users/Arthur/Desktop/ULB/2019-2020/Projet_Imagerie/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')
#print(df.iloc[4])
print(df.head(5))# Target is a binary variable: 0 if normal and 1 if pneumonia

#print(type(df))


# TRAIN IMAGES

""" Medical images are stored in a special format known as DICOM files (*.dcm). 
They contain a combination of header metadata as well as underlying raw image arrays 
for pixel data 
"""
patientId = df['patientId'][4]
dcm_file = '/Users/Arthur/Desktop/ULB/2019-2020/Projet_Imagerie/rsna-pneumonia-detection-challenge/stage_2_train_images/%s.dcm' % patientId
dcm_data = pydicom.read_file(dcm_file) # method allows to read the data of dcm_file
#print(dcm_data)
# We can see that the images have already been rescaled and resized.

im = dcm_data.pixel_array
#print(type(im))
#print(im.dtype)
#print(im.shape)

pylab.imshow(im, cmap=pylab.cm.gist_gray)
#pylab.axis('off')

# Exploring datas and labels :

def parse_data(df):
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
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]#Lambda functions is a short function in one line

    parsed = {} #dictionnary
    for n, row in df.iterrows():#Iterrate over dataframe rows
        # --- Initialize patient entry into parsed 
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': '/Users/Arthur/Desktop/ULB/2019-2020/Projet_Imagerie/rsna-pneumonia-detection-challenge/stage_2_train_images/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed


parsed = parse_data(df)

print(parsed[patientId])

# VISUALIZING TH BOXES ON THE DICOM IMAGES

def draw(data):
    """
    Method to draw single patient with bounding box(es) if present 

    """
    # --- Open DICOM file
    d = pydicom.read_file(data['dicom'])
    im = d.pixel_array

    # --- Convert from single-channel grayscale to 3-channel RGB
    im = np.stack([im] * 3, axis=2) #np.stack(): Join a sequence of arrays along a new axis.



    # --- Add boxes with random color if present
    for box in data['boxes']:
        rgb = np.floor(np.random.rand(3)*256).astype('int')# 3=size of array; juste histoire d'avoir 3 couleurs pour R, G and B
#        print(rgb)
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)

    pylab.imshow(im, cmap=pylab.cm.gist_gray)
#    pylab.axis('off')

def overlay_box(im, box, rgb, stroke=1): # What represent stroke?
    """
    Method to overlay single box on image

    """
    # --- Convert coordinates to integers
    box = [int(b) for b in box] #ok
    
    # --- Extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width
    
    print(x1,y1,x2,y2)

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im

draw(parsed[patientId])

























