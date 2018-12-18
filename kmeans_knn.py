'''

Entrenar un  modelo de KNN para clasificar la data cluserizada
con Kmeans en la clase de 11 de diciembre de 2018

17/12/2018
'''

import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

def main():
    print ("Welcome to KNN")
    # Leer el fichero de datos
    datos = np.loadtxt(open('Mall_Customers_Class1.csv', 'rb'), delimiter=',')
    ejemplos, features = np.shape(datos)
    x_total = datos[:, 0:features - 1]
    y_total = datos[:, -1]
    # Using the whole dataset for training but randomizing
    x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=0.0, random_state=2)
    # print(np.shape(x_total), np.shape(x_train))
    # print(x_total, '\n*********\n')
    # print(x_train, '\n*********\n')
    # print(y_train)
    xtest = np.loadtxt(open('xtest.csv', 'rb'), delimiter=',')
    ytest = np.loadtxt(open('ytest.csv', 'rb'), delimiter=',')
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(x_train, y_train)
    ypred = knn.predict(xtest)
    # probs = knn.predict_proba(xtest)
    # print("Probability", probs)
    target_names = ['class 0', 'class 1']
    print("Classification Report: \n", classification_report(ytest, ypred, target_names= target_names))
    cm = confusion_matrix(ytest, ypred)
    print("Confusion Matrix: \n", cm)
    P = cm[0, 0] + cm[0, 1]  # Total de Positivos
    N = cm[1, 0] + cm[1, 1]  # Total de Negativos
    TP = cm[0, 0]
    FN = cm[0, 1]
    FP = cm[1, 0]
    TN = cm[1, 1]
    error = (FP + FN) / (P + N)
    accuracy = (TP + TN) / (P + N)
    sensitivity = TP / P
    specificity = TN / N

    print('Error = ', error)
    print('Accuracy = ', accuracy)
    print('Sensitivity = ', sensitivity)
    print('Specificity = ', specificity)


if __name__ == '__main__':
    main()


'''
References:

https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

'''