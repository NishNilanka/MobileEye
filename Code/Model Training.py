#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


import os.path
import time
import numpy as np
import scipy.io as sio
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# In[3]:


def load_coordinates(filename, silent = False):
    try:
        if not silent:
            print('\tReading coordinates from %s...' % filename)
        coordinates = pd.read_csv(filename)
    except:
        print('\tFailed to read the coordinates file "%s"!' % filename)
        return None
    return coordinates


# In[4]:


import csv
import os
import re

class EyeTrackerData:
    def __init__(self, data_path):
        self.data_path = data_path
        
        meta_data = os.path.join(data_path, 'Coordinates_in_inches.csv')
        if meta_data is None or not os.path.isfile(meta_data):
            raise RuntimeError('There is no such file %s! Provide a valid dataset path.' % meta_data)
        self.meta_data = load_coordinates(meta_data)
        if self.meta_data is None:
            raise RuntimeError('Could not read metadata file %s! Provide a valid dataset path.' % meta_data)


    def __getitem__(self):
        return self.meta_data
        #im_face_path = os.path.join(self.data_path, '%05d/%05d.jpg' % (\n",
         #   self.meta_data['labelRecNum'][index], self.meta_data['frameIndex'][index]))\n",
    def __len__(self):
        return len(self.meta_data)

def atoi(text):
    return int(text) if text.isdigit() else text    

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
    
def rename_folder(path):
    filenames = next(walk(path), (None, None, []))[2]
    filenames.sort(key=natural_keys)
    count = len(filenames)
    i = 0
    for x in filenames:
        src = path + '/' +x 
        dst = path + '/image_' + str(i) + '.jpg'
        i += 1
        os.rename(src, dst)
    print(i, count)

    
dataTrain = EyeTrackerData('Filtered_data/')


videos = []

'''
meta = dataTrain.__getitem__()
for index, row in meta.iterrows():
    filePath = 'Filtered_data/' +row['sid']+ '/image_' + str(row['vid']) + '.jpg'
    isExist = os.path.exists(filePath)
    if not isExist:
        path = 'Filtered_data/' + row['sid']
        if path not in videos:
            videos.append(path)
        #rename_folder(path)
        #print('\n')
        
for x in videos:
    rename_folder(x)
'''


# In[5]:


gaze_data = []
imFace_data = []

meta = dataTrain.__getitem__()
for index, row in meta.iterrows():
    filePath = 'Filtered_data/' +row['sid']+ '/image_' + str(row['index']) + '.png'
    isExist = os.path.exists(filePath)
    if not isExist:
        print(filePath)
    else:
        gaze_data.append([row['x_inches'], row['y_inches']])
        imFace_data.append(filePath)

filepaths = pd.Series(imFace_data, name='Filepath').astype(str)
gaze_info = pd.DataFrame(gaze_data, columns=['Gazex', 'Gazey'], dtype=float)


# In[6]:


image_df = pd.concat([filepaths, gaze_info], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
train_df, test_df = train_test_split(image_df, train_size=0.7, shuffle=True, random_state=1)


# In[7]:


tf.keras.preprocessing.image.ImageDataGenerator
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255
)


# In[8]:


train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col=['Gazex', 'Gazey'],
    target_size=(128, 128),
    color_mode='grayscale',
    class_mode='raw',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col=['Gazex', 'Gazey'],
    target_size=(128, 128),
    color_mode='grayscale',
    class_mode='raw',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col=['Gazex', 'Gazey'],
    target_size=(128, 128),
    color_mode='grayscale',
    class_mode='raw',
    batch_size=32,
    shuffle=False
)


# In[22]:


def mobilnet_block(x, filters, strides):
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    return x

# stem of the model
inputs = tf.keras.Input(shape=(128, 128, 1))
x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
# main part of the model
x = mobilnet_block(x, filters=64, strides=1)
x = mobilnet_block(x, filters=128, strides=2)
x = mobilnet_block(x, filters=128, strides=1)
x = mobilnet_block(x, filters=256, strides=2)
x = mobilnet_block(x, filters=256, strides=1)
x = mobilnet_block(x, filters=512, strides=2)
for _ in range(5):
    x = mobilnet_block(x, filters=512, strides=1)
x = mobilnet_block(x, filters=1024, strides=2)
x = mobilnet_block(x, filters=1024, strides=1)
x = tf.keras.layers.AvgPool2D(pool_size=7, strides=1, padding='same')(x)
x = tf.keras.layers.Flatten()(x)


# Add GRU layer
x = tf.keras.layers.Reshape((32 * 32 , 16,))(x)
#x = tf.keras.layers.LSTM(16, return_sequences=False)(x)
x = tf.keras.layers.GRU(16, return_sequences=False)(x)
#x = tf.keras.layers.Reshape((32 * 32 , 16,))(x)
#x = tf.keras.layers.LSTM(16, return_sequences=False)(x)
#model.add(tf.keras.layers.LSTM(16, return_sequences=False))
x = tf.keras.layers.Flatten()(x)
# Add output layer for regression
#model.add(tf.keras.layers.Dense(2, activation='linear'))

outputs = tf.keras.layers.Dense(units=2, activation='linear')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='mse'
)
model.summary()


# ## CNN Model

# In[60]:


def create_convolution_layers(input_img):
    model_conv = tf.keras.layers.Conv2D(96, (11, 11), padding='same')(input_img)
    model_conv = tf.keras.layers.ReLU()(model_conv)
    model_conv = tf.keras.layers.BatchNormalization()(model_conv)
    model_conv = tf.keras.layers.MaxPooling2D((3, 3), padding='same')(model_conv)

    model_conv = tf.keras.layers.Conv2D(256, (5, 5), padding='same')(model_conv)
    model_conv = tf.keras.layers.ReLU()(model_conv)
    model_conv = tf.keras.layers.BatchNormalization()(model_conv)
    model_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(model_conv)

    model_conv = tf.keras.layers.Conv2D(384, (3, 3), padding='same')(model_conv)
    model_conv = tf.keras.layers.ReLU()(model_conv)
    model_conv = tf.keras.layers.BatchNormalization()(model_conv)

    model_conv = tf.keras.layers.Conv2D(384, (3, 3), padding='same')(model_conv)
    model_conv = tf.keras.layers.ReLU()(model_conv)
    model_conv = tf.keras.layers.BatchNormalization()(model_conv)

    model_conv = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(model_conv)
    model_conv = tf.keras.layers.ReLU()(model_conv)
    model_conv = tf.keras.layers.BatchNormalization()(model_conv)
    model_conv = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), padding='same')(model_conv)

    model_conv = tf.keras.layers.Flatten()(model_conv)
    model_conv = tf.keras.layers.Dense(4096)(model_conv)
    model_conv = tf.keras.layers.ReLU()(model_conv)
    model_conv = tf.keras.layers.Dropout(0.5)(model_conv)

    return model_conv


inputs = tf.keras.Input(shape=(128, 128, 1))
conv = create_convolution_layers(inputs)

dense = tf.keras.layers.Dense(4096)(conv)
dense = tf.keras.layers.ReLU()(dense)
dense = tf.keras.layers.Dropout(0.5)(dense)

outputs = tf.keras.layers.Dense(2, activation='linear')(dense)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='mse'
)
model.summary()


# ## CNN + LSTM

# In[24]:


def mobilnet_block(filters, strides):
    model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

# Define model
model = tf.keras.Sequential()

# Add convolutional layer
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))

model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.ReLU())

mobilnet_block(filters=64, strides=1)
mobilnet_block(filters=128, strides=2)
mobilnet_block(filters=128, strides=1)
mobilnet_block(filters=256, strides=2)
mobilnet_block(filters=256, strides=1)
mobilnet_block(filters=512, strides=2)
for _ in range(5):
    mobilnet_block(filters=512, strides=1)
mobilnet_block(filters=1024, strides=2)
mobilnet_block(filters=1024, strides=1)


model.add(tf.keras.layers.AvgPool2D((2, 2)))

# Add GRU layer
model.add(tf.keras.layers.Reshape((32 * 32 , 16,)))
model.add(tf.keras.layers.LSTM(16, return_sequences=False))
model.add(tf.keras.layers.Flatten())
# Add output layer for regression
model.add(tf.keras.layers.Dense(2, activation='linear'))

print(model)

# Compile the model
model.compile(
    optimizer='adam',
    loss='mse'
)

model.summary()


# In[36]:


import tensorflow as tf

# Load the saved model
loaded_model = tf.keras.models.load_model('saved_model/CNN_GRU_RANDOM')

converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()




# Save the quantized model to a file
with open("CNN_GRU_RANDOM_8bit.tflite", "wb") as f:
    f.write(tflite_quant_model)


# In[18]:


# Convert the model to a quantized version with 16-bit quantization
converter.target_spec.supported_types = [tf.float16]
quantized_tflite_model_16bit = converter.convert()

# Save the quantized model to a file
with open("CNN_LSTM_RANDOM_16bit.tflite", "wb") as f:
    f.write(quantized_tflite_model_16bit)


# In[26]:


import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="CNN_LSTM_RANDOM_8bit.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check the quantization type of input and output tensors
input_quantization = input_details[0]['quantization']
output_quantization = output_details[0]['quantization']

print("Input quantization:", input_quantization)
print("Output quantization:", output_quantization)


# ## CNN + LSTM weight clipping

# In[32]:


import tensorflow as tf

# Define a function for clipping weights
def clip_weights(model, clip_min=-0.5, clip_max=0.5):
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            clipped_weights = tf.clip_by_value(layer.kernel, clip_min, clip_max)
            layer.kernel.assign(clipped_weights)

# Define the mobile block with clipping
def mobilnet_block(model, filters, strides):
    model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    
    # No need to call clip_weights here

# Define model
model = tf.keras.Sequential()

# Add convolutional layer
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))

model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.ReLU())

mobilnet_block(model, filters=64, strides=1)
mobilnet_block(model, filters=128, strides=2)
mobilnet_block(model, filters=128, strides=1)
mobilnet_block(model, filters=256, strides=2)
mobilnet_block(model, filters=256, strides=1)
mobilnet_block(model, filters=512, strides=2)
for _ in range(5):
    mobilnet_block(model, filters=512, strides=1)
mobilnet_block(model, filters=1024, strides=2)
mobilnet_block(model, filters=1024, strides=1)

# Clip the weights after adding all layers
clip_weights(model)

model.add(tf.keras.layers.AvgPool2D((2, 2)))

# Add GRU layer
model.add(tf.keras.layers.Reshape((32 * 32 , 16,)))
#model.add(tf.keras.layers.LSTM(16, return_sequences=False))
model.add(tf.keras.layers.GRU(16, return_sequences=False))
#x = tf.keras.layers.GRU(16, return_sequences=False)(x)
model.add(tf.keras.layers.Flatten())
# Add output layer for regression
model.add(tf.keras.layers.Dense(2, activation='linear'))

print(model)

# Compile the model
model.compile(
    optimizer='adam',
    loss='mse'
)

model.summary()


# ## Training

# In[40]:


root_logdir = os.path.join(os.curdir, "logs\\fit\\")

def get_run_logdir():
    run_id = time.strftime("CNN_LAREG_LSTM_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=60,
    verbose=1,
    callbacks=[
        tensorboard_cb
    ]
)
#model.save('saved_model/CAPSULE_RANDOM_28_8_23')

model.save('saved_model/CNN_LAREG_LSTM_.h5')
model.save('saved_model/CNN_LAREG_LSTM_')


# ## Testing 

# In[41]:


#model = tf.keras.models.load_model('saved_model/CAPSULE_RANDOM_28_8_23.h5', custom_objects={'PrimaryCapsuleLayer': PrimaryCapsuleLayer})

# Load the model
model = tf.keras.models.load_model('saved_model/CNN_LAREG_LSTM_')

# Print model summary
model.summary()


predicted_ages = np.squeeze(model.predict(test_images))
true_ages = test_images.labels

rmse = np.sqrt(model.evaluate(test_images, verbose=0))
print("Test RMSE: {:.5f}".format(rmse))

r2 = r2_score(true_ages, predicted_ages)
print("Test R^2 Score: {:.5f}".format(r2))


# In[12]:


# Load the model
model = tf.keras.models.load_model('saved_model/CNN_LSTM_RANDOM')

# Print model summary
model.summary()


predicted_ages = np.squeeze(model.predict(test_images))
true_ages = test_images.labels

rmse = np.sqrt(model.evaluate(test_images, verbose=0))
print("Test RMSE: {:.5f}".format(rmse))

r2 = r2_score(true_ages, predicted_ages)
print("Test R^2 Score: {:.5f}".format(r2))


# # Large CNN model

# In[8]:


import tensorflow as tf

def larger_cnn_model(inputs):
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Add more convolutional blocks
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Add more convolutional blocks
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    for _ in range(15):
        x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Add fully connected layers
    x = tf.keras.layers.Dense(units=1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(units=2, activation='linear')(x)

    return outputs

# Input layer
inputs = tf.keras.Input(shape=(128, 128, 1))

# Call the larger_cnn_model function with the input layer
outputs = larger_cnn_model(inputs)

# Create the model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='mse'
)

model.summary()


# In[37]:


import tensorflow as tf

def larger_cnn_lstm_model(inputs):
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Add more convolutional blocks
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Add more convolutional blocks
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    for _ in range(5):
        x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Add LSTM layers
    x = tf.keras.layers.Reshape((8 , 8 * 1024))(x)
    x = tf.keras.layers.LSTM(16, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(16, return_sequences=False)(x)

    # Add fully connected layers
    x = tf.keras.layers.Dense(units=1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(units=2, activation='linear')(x)

    return outputs

# Input layer
inputs = tf.keras.Input(shape=(128, 128, 1))

# Call the larger_cnn_lstm_model function with the input layer
outputs = larger_cnn_lstm_model(inputs)

# Create the model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='mse'
)

model.summary()


# # Laregr Mobile Net + LSTM

# In[39]:


def mobilnet_block(filters, strides):
    model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

# Define model
model = tf.keras.Sequential()

# Add convolutional layer
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))

model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.ReLU())

mobilnet_block(filters=64, strides=1)
mobilnet_block(filters=128, strides=2)
mobilnet_block(filters=128, strides=1)
mobilnet_block(filters=256, strides=2)
mobilnet_block(filters=256, strides=1)
mobilnet_block(filters=512, strides=2)
for _ in range(15):
    mobilnet_block(filters=512, strides=1)
mobilnet_block(filters=1024, strides=2)
mobilnet_block(filters=1024, strides=1)


model.add(tf.keras.layers.AvgPool2D((2, 2)))

# Add GRU layer
model.add(tf.keras.layers.Reshape((32 * 32 , 16,)))
model.add(tf.keras.layers.LSTM(16, return_sequences=False))
model.add(tf.keras.layers.Flatten())
# Add output layer for regression
model.add(tf.keras.layers.Dense(2, activation='linear'))

print(model)

# Compile the model
model.compile(
    optimizer='adam',
    loss='mse'
)

model.summary()

