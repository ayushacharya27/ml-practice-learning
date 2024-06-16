from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
data = pd.read_csv(r"C:\Users\DELL\Desktop\ml-practice\car_data_set.csv")
data=data.dropna()
data_final = pd.get_dummies(data)
x = data_final.drop([ 'S.No.','Price'], axis=1)



 
y=data_final['Price']
print(len(x),len(y))
X_train ,  X_test,y_train , y_test= train_test_split(x,y,test_size=0.3, random_state=42)
print(len(X_train),len(y_train))

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(150,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(150,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer = 'adam',  loss='mean_squared_error', metrics=['mean_absolute_error'] )

model.fit(X_train,y_train,epochs = 3)

test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")





 
 