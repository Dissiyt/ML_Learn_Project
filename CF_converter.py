#The model will learn to convert Celsius to Fahrenheit with the formula: f= c x 1.8 + 32
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#tf.logging.set_verbosity(tf.logging.ERROR)



celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

#for i,c in enumerate(celsius_q):
   #sprint("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

#Features the inputs to our model
#Labels the outputs our model predicts
#Example A pair of inputs/outputs used during training

#create the model. We will use the simplest possible model, a Dense network. 
#Since the problem is straightforward, this network will require only a single layer, with a single neuron

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 1, input_shape = [1])
])

model.compile(loss = 'mean_squared_error',
              optimizer = tf.keras.optimizers.Adam(0.1))

history = model.fit(celsius_q, fahrenheit_a, epochs = 500, verbose = False)
print("Finished training the moedl")
print(model.predict([100.0]))

plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])



