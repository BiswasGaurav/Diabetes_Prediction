import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv("C:/Users/ob/OneDrive/Desktop/ML/Diabetics_Prediction/diabetes.csv")
print(diabetes_dataset.head())
print(diabetes_dataset.shape) #768 is the number of people
#getting the Statistical measures of the data
print(diabetes_dataset.describe())
print(diabetes_dataset['Outcome'].value_counts())#0 -> Non Diabeteic , 1 -> Diabeteic
print(diabetes_dataset.groupby('Outcome').mean())
#Separating Data and Levels

X = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

#Data Standardization
scaler = StandardScaler()

scaler.fit(X)
standardized_data = scaler.transform(X)

print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']

print(X)

X_train , X_test , Y_train , Y_test = train_test_split(X,Y , test_size = 0.2 , stratify=Y , random_state = 2)
print(X.shape , X_train.shape , X_test.shape)

#Training The model
classifier = svm.SVC(kernel="linear")

#Training the SVM to classifier
classifier.fit(X_train , Y_train)

#Model Evaluation
#Accuracy Score
# X_train_prediction = classifier.predict(X_train)
# training_data_accuracy = accuracy_score(X_train_prediction , Y_train)
# print("Accuracy Score of the Model is : " , training_data_accuracy).

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction , Y_test)
print("Accuracy Score of the Model is : " ,test_data_accuracy)
