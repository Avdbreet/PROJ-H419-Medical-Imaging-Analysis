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
df_train_tot = df_train_tot.groupby('class').apply(lambda x: x.sample(26000
                                         //3, replace=False)).reset_index(drop=True)
#Splitting the Training DataFrame into two subdataframes, for training and validation
df_train, df_valid = train_test_split(df_train_tot, test_size=0.25, random_state=2018,
                                    stratify=df_train_tot['class'])


#Plot of the validation and training datasets
print('Training and Validation datasets after splitting')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
df_train.groupby('class').size().plot.bar(ax=ax1)
ax1.title.set_text('Training set') #Pandas dataframe. groupby() function is used to split the data into groups based on some criteria
df_valid.groupby('class').size().plot.bar(ax=ax2)
ax2.title.set_text('Validation set')
plt.show() 


print('\nDivision of the training data set into a train dataset and a validaton dataset: ')
print(df_train.shape[0], ': training data')
print(df_valid.shape[0], ': validation data')


###############################################################################
#                                                                             #
#                           DATAGENERATOR CREATION                            #
#                                                                             #
###############################################################################
from keras.preprocessing.image import ImageDataGenerator

IMG_SIZE=(128,128)
BATCH_SIZE=32

train_images_dir= base_dir+'stage_2_images_jpeg/'

###############################################################################
#                          Training image generator                           #
###############################################################################

#data agmentation
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


print('\nPlot of the augmented images:')

def plotImages(images_arr,row=2,col=4,fig_size=(16,8)):
    fig, axes = plt.subplots(row, col, figsize=fig_size)
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

augmented_images = [train_generator[0][0][0] for i in range(8)]
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
#                                                                             #
#                             DNN model building                              #
#                                                                             #
###############################################################################

from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

input_shape=(128,128,3)
kernel_size=3
lr=1e-4

###############################################################################
#                                 Architecture                                #
###############################################################################
model=Sequential()

model.add(Conv2D(64, kernel_size, padding='same', activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size, padding='same', activation='relu'))
model.add(MaxPool2D(2))

model.add(Conv2D(128, kernel_size, padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size, padding='same', activation='relu'))
model.add(MaxPool2D(2))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size, padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size, padding='same', activation='relu'))
model.add(MaxPool2D(2))
model.add(Dropout(0.25))

model.add(Conv2D(512, kernel_size, padding='same', activation='relu'))
model.add(Conv2D(512, kernel_size, padding='same', activation='relu'))
model.add(MaxPool2D(2))
model.add(Dropout(0.25))

model.add(GlobalAveragePooling2D())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(3, activation = 'softmax'))

model.summary()


###############################################################################
#                              Model compilation                              #
###############################################################################

model.compile(optimizer = Adam(lr = lr), loss = 'categorical_crossentropy',
                        metrics = ['categorical_accuracy'])


###############################################################################
#                               Model fitting                                 #
###############################################################################

weight_path="{}_weights.best.hdf5".format('lung_opacity')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=10) 
callbacks_list = [early,checkpoint]

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size

training=model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validation_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=40, callbacks=callbacks_list)


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

"""To modify, make it more beautiful!"""

model.load_weights(weight_path)
model.save('full_model.h5')


"""If working, delete test generator"""

val_loss, val_acc=model.evaluate_generator(generator=validation_generator,
steps=len(validation_generator))

print('Validation Loss:%f' %val_loss)
print('Validation Accuracy:%f'%val_acc)


###############################################################################
#                        Predition on validation set                          #
###############################################################################

#Input and actual output of a big sample from validation datas
valid_x, valid_y=next(image_gen_val.flow_from_dataframe(dataframe=df_valid, directory=train_images_dir, x_col='path', y_col='class',
                                                  seed=42, shuffle=True, target_size=IMG_SIZE,
                                                  class_mode='categorical', batch_size=6000))

#Predicted output from the model for the same  dataset
pred_y=model.predict(valid_x,batch_size=BATCH_SIZE, verbose=True)

###############################################################################
#                            Performance measures                             #
###############################################################################

from sklearn.metrics import  classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle

predicted_label=np.argmax(pred_y, axis=1)
actual_label=np.argmax(valid_y, axis=1)

#Confusion Matrix:
c_matrix=confusion_matrix(actual_label,predicted_label)
print('\n Confusion Matrix')
print(c_matrix)
plt.matshow(c_matrix)
plt.show()

#Classification report:
print('\n Classification report')
print(classification_report(actual_label,predicted_label, target_names=class_enc.classes_))

#ROC curves
#Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

fpr=dict()
tpr=dict()
roc_auc=dict()

n_classes=3
for i in range(n_classes):
    fpr[i],tpr[i],_=roc_curve(valid_y[:,i], pred_y[:,i])
    roc_auc[i]=auc(fpr[i],tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(valid_y.ravel(), pred_y.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

# Plot all ROC curves
print('\n ROC curves')
plt.figure()
lw=1.5
plt.plot(fpr["micro"], tpr["micro"],
         label='Average - AUC = {0:0.2f}'
               ''.format(roc_auc["micro"]),
         color='red', linestyle=':', linewidth=4)

colors = cycle(['royalblue', 'darkorange', 'limegreen'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='Class {0} - AUC = {1:0.2f}'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw, label='Random guessing')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
















