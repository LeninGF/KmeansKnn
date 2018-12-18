'''

Clustering data with Kmeans.
Data to be used is Mall_Customers2.csv

17/12/2018
'''

import numpy as np
from sklearn.cluster import KMeans
from random import randint

def generate_data(examples, features):
    min_age = 18
    max_age = 70
    min_income = 15
    max_income = 140
    min_spending_score = 1
    max_spending_score = 100
    min_gender = 1
    max_gender = 2

    x = np.zeros((examples,features))
    for i in range(examples):
        for j in range(features):
            if j==0:
                tmp = randint(min_gender,max_gender)
            elif j==1:
                tmp = randint(min_age, max_age)
            elif j==2:
                tmp = randint(min_income, max_income)
            elif j==3:
                tmp = randint(min_spending_score, max_spending_score)
            x[i,j] = tmp
    return x

def main():
    print("Welcome to KNN")
    # Leer el fichero de datos
    datos = np.loadtxt(open('Mall_Customers2.csv', 'rb'), delimiter=',')
    ejemplos, features = np.shape(datos)
    X = datos
    kmeans = KMeans(n_clusters=2, random_state=23).fit(X)
    print(kmeans.labels_)
    print("Centers, \n", kmeans.cluster_centers_)
    print(np.shape(kmeans.labels_))
    np.savetxt('labels.csv', kmeans.labels_, delimiter=',')

    xtest = generate_data(60,4)
    print(xtest)
    ytest = kmeans.predict(xtest)
    print(ytest)
    np.savetxt('xtest.csv', xtest, delimiter=',')
    np.savetxt('ytest.csv', ytest, delimiter=',')
if __name__ == '__main__':
    main()

