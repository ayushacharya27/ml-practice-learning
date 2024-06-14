from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df1=pd.read_csv("C:\\Users\\DELL\\Desktop\\data_set_building\\Housing.csv")
#df1.replace({'yes': 1, 'no': 0}, inplace=True)
df=pd.get_dummies(df1)
X = df.drop('price',axis=1)   
y = df['price'] 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=None, shuffle=True, stratify=None)

model = LinearRegression()
model.fit(X_train,y_train)

'''pred = model.predict(X_test)
accuracy = accuracy_score(y_test,pred)
print(accuracy)'''
y_pred = model.predict(X_test)

 
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R-squared (R2): {r2}')
print(f'Mean Squared Error (MSE): {mse}')
