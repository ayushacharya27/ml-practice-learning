import tensorflow as tf
mnist = tf.keras.datasets.mnist
(X_train ,y_train) , (X_test,y_test)=mnist.load_data()

model = tf.keras.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(150, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(170 , activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=10)


val_loss, val_accuracy = model.evaluate(X_test, y_test)
print(val_loss, val_accuracy)
model.save('mnist_neural_network.model')
