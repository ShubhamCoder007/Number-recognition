import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist    # 28x28 image of handwritten digits 0-9

(x_train,y_train), (x_test,y_test) = mnist.load_data()

#Scaling or normalizing the datasets
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

#Creating the architecture of the network
#Sequential feed forward NN
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())  #28x28 has to be flattened; can also use numpy reshape;   input layer
model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))   # relu: rectified linear activation function
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))  
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax)) # output layer, neurons is the number of classification
#softmax output as probability distribution

#optimizer are the cost function minimization routine such as gradient descend
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 5)


#output after the execution:
#Epoch 1/5
#60000/60000 [==============================] - 9s 147us/step - loss: 0.2621 - acc: 0.9237
#Epoch 2/5
#60000/60000 [==============================] - 7s 118us/step - loss: 0.1083 - acc: 0.9667
#Epoch 3/5
#60000/60000 [==============================] - 7s 118us/step - loss: 0.0737 - acc: 0.9772
#Epoch 4/5
#60000/60000 [==============================] - 7s 119us/step - loss: 0.0547 - acc: 0.9826
#Epoch 5/5
#60000/60000 [==============================] - 7s 121us/step - loss: 0.0406 - acc: 0.9867
#<tensorflow.python.keras._impl.keras.callbacks.History at 0x212922c7f98>

#validating data
val_loss, val_acc = model.evaluate(x_test, y_test)
print("loss: %s accuracy: %s"%(val_loss*100,val_acc*100))

#output:
#10000/10000 [==============================] - 1s 83us/step
#loss: 9.442256451789289 accuracy: 97.09


plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()
#print(x_train[0])

#Saving the model
model.save('num_recog.model')

#reloading the model for the new session
#new_model = tf.keras.models.load_model('num_recog.model')

#predicting the results
predictions = model.predict([x_test])
#print(predictions)

#according to probabilistic approach desired output is given by argmax
print(np.argmax(predictions[0]))

#Data for which the result is shown
plt.imshow(x_test[0]) #cmap = plt.cm.binary)
plt.show()

#more test predictions
print("\n%s"%np.argmax(predictions[5]))
print("\n%s"%np.argmax(predictions[10]))

plt.imshow(x_test[5])
plt.show()

plt.imshow(x_test[10])
plt.show()


