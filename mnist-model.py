from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import joblib

 
mnist = fetch_openml("mnist_784")
x = mnist['data']
y = mnist['target']

 
x_train = x[:60000]
y_train = y[:60000]
x_test = x[60000:]
y_test = y[60000:]

 
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

joblib.dump(model,"mnist.model.pkl")

single_sample = x_test.iloc[0].values.reshape(1, -1)
y_pred_single = model.predict(single_sample)
print(y_pred_single)
print(y_test.iloc[0])



 
