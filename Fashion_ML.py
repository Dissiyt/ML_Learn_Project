#from_future_import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm

#tf.enable_eager_execution()

#Define and get dataset
dataset, metadata = tfds.load('fashion_mnist', as_supervised = True, with_info = True)
train_dataset, test_dataset = dataset['train'], dataset['test']

#Array to label each image type in the dataset
class_names = metadata.features['label'].names

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

#define normalize function to asign to train and test dataset
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

#The map function applies the normalize function to each element in the train and test dataset
tain_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

#Cache the dataset
train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()

#Display a image and 25
#for image, label in test_dataset.take(1):
    #break
#image = image.numpy().reshape((28, 28))
#plt.figure()
#plt.imshow(image, cmap=plt.cm.binary)
#plt.colorbar()
#plt.grid(False)
#plt.show()
#plt.figure(figsize=(10,10))
#for i, (image, label) in enumerate(train_dataset.take(25)):
    #image = image.numpy().reshape((28,28))
    #plt.subplot(5,5,i+1)
    #plt.xticks([])
    #plt.yticks([])
    #plt.grid(False)
    #plt.imshow(image, cmap=plt.cm.binary)
    #plt.xlabel(class_names[label])
#plt.show()

#Set up the Layers
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28, 1)),
    tf.keras.layers.Dense(128, activation = tf.nn.relu),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])

#compile the model
model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])

BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)
model.fit(train_dataset, epochs = 5, steps_per_epoch = math.ceil(num_train_examples/BATCH_SIZE))

#Evaluate Accuracy
test_loss, test_accuracy = model.evaluate(test_dataset, steps = math.ceil(num_test_examples/32))
print('Accuracy on test dataset:' , test_accuracy)

#Make predictions
for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)
predictions.shape
predictions[0]
np.argmax(predictions[0])
test_labels[0]

#Graph the predictions
def plot_image(i, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img[...,0], cmap = plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
                                         color = color)
    
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = "#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()
