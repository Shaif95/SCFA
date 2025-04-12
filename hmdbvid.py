import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
import json
import glob
import random
import collections
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from sklearn.utils import shuffle

from skimage.io import imread
from skimage.transform import resize
from keras import layers
from keras import models
import keras
from keras.layers import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import shutil
import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import keras
import json
import tensorflow as tf 
from keras.layers import Input
from keras import Sequential
from keras.layers import Dense, LSTM,Flatten, TimeDistributed, Conv2D, Dropout
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.models import load_model
from keras.callbacks import Callback,ModelCheckpoint
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K
import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.transform import resize
from keras import layers
from keras import models
import keras
from keras.layers import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv1D,Reshape, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten, UpSampling2D
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.transform import resize
from keras import layers
from keras import models
import keras
from keras.layers import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.callbacks import Callback,ModelCheckpoint
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K


AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 32
EPOCHS = 50
CROP_TO = 32
SEED = 26

PROJECT_DIM = 128
LATENT_DIM = 512
WEIGHT_DECAY = 0.0005
learning_rate = 0.0001
batch_size = 32
hidden_units = 512
projection_units = 256
num_epochs = 2
dropout_rate = 0.5

temperature = 0.05

def get_f1(y_true, y_pred): #taken from old keras source code
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    tn = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1-y_true) * (y_pred), 0, 1)))
    fn = K.sum(K.round(K.clip((y_true) * (1-y_pred), 0, 1)))
    

    f1_val = tp / ( tp + ( (1/2) * (fp+fn) ) + K.epsilon())
    return f1_val

from glob import glob
trn='D:/D-Video/hmdb/*/'
tr= glob(trn)

len(tr)

import tensorflow as tf

# Explicitly set device to CPU
#tf.config.set_visible_devices([], 'GPU')

# Now TensorFlow will use CPU instead of GPU
import numpy as np
from glob import glob

label = 0
XT = []
YT = []
for i in tr:
    lst = glob(i + "/*")
    for j in lst:
        imgs = glob(j + "/*")
        if imgs:  # Only append non-empty image lists
            XT.append(imgs)
            YT.append(label)
    label += 1

print("Number of samples:", len(XT))
print("Number of labels:", len(YT))
if XT:
    print("Shape of first sample:", np.shape(XT[0]))
    print("First label:", YT[0])
else:
    print("XT is empty.")


i = 9
x = glob(tr[i]+'/*/')
vid = glob(x[i] +'/*')
vid[:2]


from PIL import Image
import numpy as np

def resize_images(q):
    img = Image.fromarray(q)
    resized_img = img.resize((56,56), Image.ANTIALIAS)
    return resized_img

def generate_and_sort(n):
    unique_numbers = set()
    while len(unique_numbers) < 16:
        unique_numbers.add(random.randint(0, n))
    sorted_numbers = sorted(unique_numbers)
    return sorted_numbers


def prepare_videoes(image_paths):
    images = []
    for path in image_paths:
        # Load image
        img = Image.open(path)
        
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize the image
        img = img.resize((56,56), Image.ANTIALIAS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Append the processed image to the list
        images.append(img_array)
    
    n = np.shape(images)[0] - 1
    sortn = generate_and_sort(n)
    img_parts = [images[i] for i in sortn]
    
    combined_image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    for i, img_part in enumerate(img_parts):
        row = i // 4
        col = i % 4
        combined_image[row*56:(row+1)*56, col*56:(col+1)*56, :] = img_part
        
    return combined_image

combined_image = prepare_videoes(vid)
print("Combined image shape:", combined_image.shape)

q =  Image.fromarray(combined_image)
resized_image = q.resize((560,560), Image.ANTIALIAS)
# Display the resized image using matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(resized_image)
plt.axis('off')  # Turn off axis numbers and ticks
#plt.show()

BATCH_SIZE = 32
# Width and height of image
IMAGE_SIZE = 224

class My_Test_Generator(keras.utils.Sequence):
    def __init__(self, filename, batch_size):
        self.filename = filename
        self.batch_size = batch_size
        
    def __len__(self):
        return (np.ceil(len(self.filename) / float(self.batch_size))).astype(np.int32)
    
    def __getitem__(self, idx):
        batch_x = self.filename[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = batch_x  # Assuming y_train is similar to x_train for demonstration
        
        x_batch = np.array([prepare_videoes(filename) for filename in batch_x]) / 255.0
        y_batch = np.array([prepare_videoes(filename) for filename in batch_y]) / 255.0
        
        return x_batch, y_batch
    
my_generator = My_Test_Generator(XT, batch_size)

class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.1, **kwargs):
        super(SupervisedContrastiveLoss, self).__init__(**kwargs)
        self.temperature = temperature

    def call(self, y_true, y_pred):
        # Split y_pred into proj_1 and proj_2
        batch_size = tf.shape(y_pred)[0] // 2
        proj_1 = y_pred[:batch_size]
        proj_2 = y_pred[batch_size:]

        # Concatenate labels
        labels = tf.concat([y_true, y_true], axis=0)

        # Normalize projections
        projections = tf.math.l2_normalize(tf.concat([proj_1, proj_2], axis=0), axis=1)

        # Compute similarity matrix
        similarities = tf.matmul(projections, tf.transpose(projections)) / self.temperature

        # Mask out self-similarities
        mask = tf.eye(tf.shape(similarities)[0], dtype=tf.bool)
        similarities = tf.where(mask, tf.zeros_like(similarities), similarities)

        # Exponentiate similarities
        exp_similarities = tf.exp(similarities)

        # Compute denominator for contrastive loss
        denominator = tf.reduce_sum(exp_similarities, axis=1, keepdims=True)

        # Compute supervised contrastive loss
        positive_mask = tf.equal(labels[:, None], labels[None, :])
        positive_similarities = tf.where(positive_mask, exp_similarities, tf.zeros_like(exp_similarities))
        numerator = tf.reduce_sum(positive_similarities, axis=1)

        loss = -tf.reduce_mean(tf.math.log(numerator / denominator))
        return loss


from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model

def create_supervised_contrastive_model(input_shape=(224, 224, 3), projection_dim=128):
    resnet_base = ResNet50(include_top=False, weights="imagenet", pooling="avg")

    inputs_1 = layers.Input(shape=input_shape, name="input_1")
    inputs_2 = layers.Input(shape=input_shape, name="input_2")

    features_1 = resnet_base(inputs_1)
    features_2 = resnet_base(inputs_2)

    projection_1 = layers.Dense(512, activation='relu')(features_1)
    projection_1 = layers.Dense(projection_dim)(projection_1)

    projection_2 = layers.Dense(512, activation='relu')(features_2)
    projection_2 = layers.Dense(projection_dim)(projection_2)

    # Concatenate projections for supervised contrastive loss
    concatenated_projections = layers.Concatenate(axis=0)([projection_1, projection_2])

    contrastive_model = Model(inputs=[inputs_1, inputs_2], outputs=concatenated_projections)
    return contrastive_model


# Instantiate the supervised contrastive model
input_shape = (224, 224, 3)
projection_dim = 128
contrastive_model = create_supervised_contrastive_model(input_shape, projection_dim)
contrastive_model.summary()

from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=1e-4)
contrastive_loss = SupervisedContrastiveLoss()

contrastive_model.compile(optimizer=optimizer, loss=contrastive_loss)

class My_Test_Generator(tf.keras.utils.Sequence):
    def __init__(self, filename, labels, batch_size):
        self.filename = filename
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.filename) / float(self.batch_size))).astype(np.int32)

    def __getitem__(self, idx):
        batch_x = self.filename[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        x_batch = np.array([prepare_videoes(filename) for filename in batch_x]) / 255.0
        y_batch = np.array(batch_y)  # Include labels for supervised contrastive loss

        return x_batch, y_batch

# Instantiate the generator
BATCH_SIZE = 32
my_generator = My_Test_Generator(XT, YT, BATCH_SIZE)
# Training loop
epochs = 20

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    epoch_loss = 0  # Initialize loss accumulator
    batch_count = 0  # Count the number of batches

    for x_batch, y_batch in my_generator:
        with tf.GradientTape() as tape:
            # Pass inputs to the model
            y_pred = contrastive_model([x_batch, x_batch], training=True)
            # Calculate supervised contrastive loss
            loss = contrastive_loss(y_batch, y_pred)

        # Backpropagation
        gradients = tape.gradient(loss, contrastive_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, contrastive_model.trainable_variables))

        # Accumulate loss and batch count
        epoch_loss += loss.numpy()
        batch_count += 1

    # Compute average loss for the epoch
    average_loss = epoch_loss / batch_count
    print(f"Average Loss for Epoch {epoch + 1}: {average_loss}")


# Save the entire contrastive model
contrastive_model.save("hmdb_contrastive_model.h5")
print("SimCLR contrastive model saved as 'hmdb_contrastive_model.h5'")
