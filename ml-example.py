#%%
import numpy as np
import tensorflow as tf
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

IMG_WIDTH=888
IMG_HEIGHT=888
img_folder=r'picture'

plt.figure(figsize=(20,20))
test_folder=r'picture\test'

# %%
data_dir = 'picture'

#%%
batch_size = 32
img_height = 222
img_width = 108

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=1,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=1,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

#%%
num_classes = 5

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

#%%
train_history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)

# %%
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

show_train_history(train_history,'accuracy','val_accuracy')

# %%
checkImg = 'C:/imagepath/test.png'
img = tf.keras.utils.load_img(
    checkImg, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# %%
import glob
import os
images = glob.glob('temp/*.png')
specificClass = 'spec'

for fileName in images:
    img = tf.keras.utils.load_img(
                        fileName, target_size=(img_height, img_width)
                    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])   

    if class_names[np.argmax(score)] == specificClass:          
        os.rename(fileName, "done/"+os.path.basename(fileName))
