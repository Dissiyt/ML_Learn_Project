import tensorflow as tf
import datetime

# Clear any logs from previous runs


mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name='layers_flatten'),
    tf.keras.layers.Dense(512, activation='relu', name='layers_dense'),
    tf.keras.layers.Dropout(0.2, name='layers_dropout'),
    tf.keras.layers.Dense(10, activation='softmax', name='layers_dense_2')
  ])



#Sequential is useful for stacking layers where each layer has one input tensor and one output tensor. 
# Layers are functions with a known mathematical structure that can be reused and have trainable variables. 
# Most TensorFlow models are composed of layers. This model uses the Flatten, Dense, and Dropout layers.

model = create_model()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x = x_train,
          y= y_train,
          epochs=5, 
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard_callback])

#load and prepare the dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.shuffle(60000).batch(64)
test_dataset = test_dataset.batch(64)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# Define our metrics
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

#Define training test functions
def train_step(model, optimizer, x_train, y_train):
  with tf.GradientTape() as tape:
    predictions = model(x_train, training=True)
    loss = loss_object(y_train, predictions)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  train_loss(loss)
  train_accuracy(y_train, predictions)
  
def test_step(model, x_test, y_test):
  predictions = model(x_test)
  loss = loss_object(y_test, predictions)

  test_loss(loss)
  test_accuracy(y_test, predictions)
  
predictions = model(x_train[:1]).numpy()
predictions

#Set up summary writers to write the summaries to disk in a different logs directory:
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


#The tf.nn.softmax function converts these logits to probabilities for each class:
tf.nn.softmax(predictions).numpy()

#Define a loss function for training 
#The loss function takes a vector of ground truth values and a vector of logits and returns a scalar loss for each example.
#This loss is equal to the negative log probability of the true class: The loss is zero if the model is sure of the correct class.
#loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


loss_object(y_train[:1], predictions).numpy()

#model.evaluate(x_test,  y_test, verbose=2)
#train_accuracy = (y_train, predictions)
#train_loss = (loss_fn)

model = create_model() # reset our model

EPOCHS = 5

for epoch in range(EPOCHS):
  for (x_train, y_train) in train_dataset:
    train_step(model, optimizer, x_train, y_train)
  with train_summary_writer.as_default():
    tf.summary.scalar('loss', train_loss.result(), step=epoch)
    tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

  for (x_test, y_test) in test_dataset:
    test_step(model, x_test, y_test)
  with test_summary_writer.as_default():
    tf.summary.scalar('loss', test_loss.result(), step=epoch)
    tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(), 
                         train_accuracy.result()*100,
                         test_loss.result(), 
                         test_accuracy.result()*100))

  # Reset metrics every epoch
  train_loss.reset_states()
  test_loss.reset_states()
  train_accuracy.reset_states()
  test_accuracy.reset_states()





#writer = tf.summary.create_file_writer('logs')
#with writer.as_default():
  #tf.summary.scalar('loss', train_loss, step = 1)
  #tf.summary.scalar('loss',train_loss, step = 2)
  #tf.summary.scalar('loss', train_loss, step = 3)
  #tf.summary.scalar('loss', train_loss, step = 4)
  #tf.summary.scalar('loss', train_loss, step = 5)
  #tf.summary.scalar('accuracy', train_accuracy, step = 1)
  #tf.summary.scalar('accuracy', train_accuracy, step = 2)
  #tf.summary.scalar('accuracy', train_accuracy, step = 3)
  #tf.summary.scalar('accuracy', train_accuracy, step = 4)
  #tf.summary.scalar('accuracy', train_accuracy, step = 5)