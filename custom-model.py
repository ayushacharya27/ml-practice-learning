import os
import numpy as np
import os
import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score 

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):  # Check if it's a directory
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    resized_img = cv2.resize(img, (28, 28))   
                    normalized_img = resized_img / 255.0   
                    images.append(normalized_img.flatten())  
                    labels.append(int(label))   
    return np.array(images), np.array(labels)

 
train_images, train_labels = load_images_from_folder("C:\\Users\\DELL\\Desktop\\custom dataset\\train")
test_images, test_labels = load_images_from_folder('C:\\Users\\DELL\\Desktop\\custom dataset\\test')
 

model = RandomForestClassifier(n_estimators=100)
model.fit(train_images, train_labels)
predict=model.predict(test_images)
accuracy = accuracy_score(predict,test_labels)
print(accuracy)
joblib.dump(model,'custom_model.pkl')
