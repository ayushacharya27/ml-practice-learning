from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from sklearn.utils import shuffle
 


data = pd.read_csv(r"C:\Users\DELL\Desktop\used_cars_data.csv")

data= data.dropna()
data_final=pd.get_dummies(data)
data_final = shuffle(data_final, random_state=42)


x = data_final.drop([ 'S.No.','Price'], axis=1)



 
y=data_final['Price']
print(len(x))
print(len(y))
X_train , X_test, y_train , y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

 
#y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f'Accuracy score: {r2}')