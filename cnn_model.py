import pandas as pd
import numpy as np
import os
import pydicom
import glob
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


###############################################################################
#                                                                             #
#                           DATAFRAME CREATION                                #
#                                                                             #
###############################################################################

#Basis directory of all the files
base_dir='/Users/Arthur/Desktop/ULB/2019-2020/Projet_Imagerie/rsna-pneumonia-detection-challenge/'

###############################################################################
#                           Training DataFrame                                #
###############################################################################

"""
#Change of dicom to jpeg
    
inputdir = base_dir+'stage_2_train_images/'
outdir = base_dir+'stage_2_images_jpeg/'

test_list = [os.path.basename(x) for x in glob.glob(inputdir + './*.dcm')]
for f in test_list:   
    ds = pydicom.read_file( inputdir + f) # read dicom image
    img = ds.pixel_array # get image array
    img_mem = Image.fromarray(img) # Creates an image memory from an object exporting the array interface
    img_mem.save(outdir + f.replace('.dcm','.jpeg')) 
"""

#Creation of the test dataframe
df_class=pd.read_csv(base_dir+'stage_2_detailed_class_info.csv')
df_train_tot=pd.read_csv(base_dir+'stage_2_train_labels.csv')

#Add a column with the class (string)
ar_class=np.array(df_class.iloc[:,[1]])
df_train_tot['class']=ar_class

#Add a column with the paths, intitialized with empty strings
ar_path=np.array(['' for x in range(len(df_class.iloc[:,[1]]))])
df_train_tot['path']=ar_path

for i in range(len(df_train_tot)):
    Id=df_train_tot.loc[i,'patientId']
    path='%s.jpeg' %Id
    df_train_tot.loc[i,'path']= path

class_enc=LabelEncoder()
df_train_tot['class_index']=class_enc.fit_transform(df_train_tot['class'])

print('\n Head of the df_train_tot: \n')
print(df_train_tot.head(5))

#Sort training datas by classes
print('\n Train Dataset, before and after grouping in equal amount amongst the classes \n')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
df_train_tot.groupby('class').size().plot.bar(ax=ax1) #Pandas dataframe. groupby() function is used to split the data into groups based on some criteria
df_train_tot = df_train_tot.groupby('class').apply(lambda x: x.sample(26100
                                         //3, replace=False)).reset_index(drop=True)
df_train_tot.groupby('class').size().plot.bar(ax=ax2)
plt.show() 

#Splitting the Training DataFrame into two subdataframes, for training and validation
df_train, df_valid = train_test_split(df_train_tot, test_size=0.25, random_state=2018,
                                    stratify=df_train_tot['class'])

print('Division of the training data set into a train dataset and a validaton dataset:\n ')
print(df_train.shape, ': training data')
print(df_valid.shape, ': validation data')


###############################################################################
#                           Test DataFrame                                    #
###############################################################################

df_test=pd.read_csv(base_dir + "stage_2_sample_submission.csv")

#Add a column with the path, initialized with empty strings
ar_path_test=np.array(['path' for x in range(len(df_test.iloc[:,[1]]))])
df_test['path']=ar_path_test


for i in range(len(df_test)):
    Id=df_test.loc[i,'patientId']
    path='%s.jpeg' %Id
    df_test.loc[i,'path']= path

print('Head of df_test')
print(df_test.head(3))


###############################################################################
#                                                                             #
#                           DATAGENERATOR CREATION                            #
#                                                                             #
###############################################################################
from keras.preprocessing.image import ImageDataGenerator

IMG_SIZE=(128,128)
BATCH_SIZE=32

train_images_dir= base_dir+'stage_2_images_jpeg/'
test_images_dir=base_dir+'stage_2_test_images_jpeg/'

###############################################################################
#                          Training image generator                           #
###############################################################################


image_gen_train=ImageDataGenerator(rescale=1./255., samplewise_center=False, 
                              samplewise_std_normalization=False, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range = 0.05, 
                              width_shift_range = 0.02, 
                              rotation_range = 5, 
                              shear_range = 0.01,
                              zoom_range = 0.05)

print('\nTrain Generator:')
train_generator=image_gen_train.flow_from_dataframe(dataframe=df_train, directory=train_images_dir,
                                                    x_col='path',y_col='class',
                                                    batch_size=BATCH_SIZE, seed=42,shuffle=True,
                                                    class_mode='categorical', target_size=IMG_SIZE)

#output of the training generator 
train_x, train_y=next(train_generator)

print('\nOutput:')
print(train_x.shape, train_y.shape)

print('Plot of the augmented images:\n')

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

augmented_images = [train_generator[0][0][0] for i in range(5)]
plotImages(augmented_images)

###############################################################################
#                         Validation image generator                          #
###############################################################################

image_gen_val=ImageDataGenerator(rescale=1./255)

print('\nValidation Generator:')
validation_generator=image_gen_val.flow_from_dataframe(dataframe=df_valid, directory=train_images_dir, 
                                                       x_col='path', y_col='class',
                                                       batch_size=BATCH_SIZE, seed=42, shuffle=True,
                                                       class_mode='categorical',target_size=IMG_SIZE)
                                                       

###############################################################################
#                            Test image generator                             #
###############################################################################

""" To be modified """

image_gen_test=image_gen_val

print('\nTest Generator:')
test_generator=image_gen_val.flow_from_dataframe(dataframe=df_test,directory=test_images_dir,
                                                 x_col='path',
                                                 y_col='class', shuffle=False,
                                                 class_mode=None, target_size=IMG_SIZE)


###############################################################################
#                                                                             #
#                             DNN model building                              #
#                                                                             #
###############################################################################

from tensorflow.keras.layers import MaxPool2D
from keras import optimizers
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


#Architecture
model=Sequential()

model.add(Conv2D(64, 3, padding='same', activation='relu', input_shape=(128,128,3)))
model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPool2D(2))

model.add(Conv2D(128, 3, padding='same', activation='relu'))
model.add(Conv2D(128, 3, padding='same', activation='relu'))
model.add(MaxPool2D(2))
model.add(Dropout(0.25))

model.add(Conv2D(256, 3, padding='same', activation='relu'))
model.add(Conv2D(256, 3, padding='same', activation='relu'))
model.add(Conv2D(256, 3, padding='same', activation='relu'))
model.add(MaxPool2D(2))
model.add(Dropout(0.25))

model.add(Conv2D(512, 3, padding='same', activation='relu'))
model.add(Conv2D(512, 3, padding='same', activation='relu'))
model.add(Conv2D(512, 3, padding='same', activation='relu'))
model.add(MaxPool2D(2))
model.add(Dropout(0.25))

model.add(Conv2D(512, 3, padding='same', activation='relu'))
model.add(Conv2D(512, 3, padding='same', activation='relu'))
model.add(Conv2D(512, 3, padding='same', activation='relu'))
model.add(MaxPool2D(2))

model.add(GlobalAveragePooling2D())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(LeakyReLU(0.1))
model.add(Dense(3, activation = 'softmax'))

model.summary()


#Compilation
model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy',
                        metrics = ['categorical_accuracy'])


#Fitting

weight_path="{}_weights.best.hdf5".format('lung_opacity')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=10) 
callbacks_list = [early,checkpoint]

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

""" Everything run propely until here """

training=model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validation_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=20, callbacks=callbacks_list) #Crash if running


###############################################################################
#                                                                             #
#                              Interpretations                                #
#                                                                             #
###############################################################################

###############################################################################
#                        Loss and accuraccy evolution                         #
###############################################################################

acc = training.history['categorical_accuracy']
val_acc = training.history['val_categorical_accuracy']

loss=training.history['loss']
val_loss=training.history['val_loss']
epochs=15
epochs_range = range(epochs)

plt.figure(figsize=(8,8))
plt.subplot(1, 2, 1)
plt.plot( acc, label='Training Accuracy')
plt.plot( val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot( loss, label='Training Loss')
plt.plot( val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

###############################################################################
#                              Model Evalutaion                               #
###############################################################################

"""To modify!"""

model.load_weights(weight_path)
model.save('full_model.h5')

val_loss, val_acc=model.evaluate_generator(generator=validation_generator,
steps=STEP_SIZE_TEST)

print('Loss:%f' %val_loss)
print('Accuracy:%f'%val_acc)


test_loss, test_acc=model.evaluate_generator(generator=test_generator,
steps=STEP_SIZE_TEST)

print('Loss:%f' %test_loss)
print('Accuracy:%f'%test_acc)
