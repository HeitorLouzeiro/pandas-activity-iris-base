import statistics

import pandas as pd

space = '-' * 50

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'  # noqa

col_names = ['Sepal_Length', 'Sepal_Width',
             'Petal_Length', 'Petal_Width', 'Class']
iris = pd.read_csv(url, names=col_names)

print(iris)

print(space)
# 2 - Identify and Classify the Variables.
print('2 - Identify and Classify Variables.\n')

# 2.1 - Quantitative Variables
print('2.1 - Quantitative Variables')
print('Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width')


# Qualitativas
print('\n2.2 - Qualitative Variables')
print('Class')

print(space)

'''
3 - Calculate the following measurements on the
size of sepals (Sepal_Length)
'''
print('3 - Calculate the following measurements on the size of the sepals (Sepal_Length)\n')  # noqa

# average
print('Average:', statistics.mean(iris['Sepal_Length']))

# Median
print('median:', statistics.median(iris['Sepal_Length']))

# standard deviation
print('Standard Deviation:', statistics.pstdev(iris['Sepal_Length']))

# full range
print('Amplitude Total:', max(
    iris['Sepal_Length']) - min(iris['Sepal_Length']))

# coefficient of variation
print('Coeficiente de Variação:', statistics.pstdev(iris['Sepal_Length']) / statistics.mean(iris['Sepal_Length']))  # noqa
print(space, '\n')

'''
4 - Separate the data by 'Class' and check:
a) Which class has the largest petal size on average?
b) which class has the smallest sepal width on average?
c) rank each class in order of homogeneity
'''

# 4 - Separate the data by 'Class' and check:
irisClass = iris['Class'].unique()
print('Classes:', irisClass)
print('\n', space)

irisSetosa = iris[iris['Class'] == 'Iris-setosa']
print(irisSetosa.head(5))
print(space)

irisVersicolor = iris[iris['Class'] == 'Iris-versicolor']
print(irisVersicolor.head(5))
print(space)

irisVirginica = iris[iris['Class'] == 'Iris-virginica']
print(irisVirginica.head(5))
print(space)


# a) Which class has the largest petal size on average?
irisSetosaPetalLength = statistics.mean(irisSetosa['Petal_Length'])
irisVersicolorPetalLength = statistics.mean(irisVersicolor['Petal_Length'])
irisVirginicaPetalLength = statistics.mean(irisVirginica['Petal_Length'])

print('Average Petal_Length Iris-setosa:', irisSetosaPetalLength)
print('Average Petal_Length Iris-versicolor:', irisVersicolorPetalLength)
print('Average Petal_Length Iris-virginica:', irisVirginicaPetalLength)

print('\na) Which class has the largest petal size on average? Iris-virginica')
print(space)

# b) which class has the smallest sepal width on average?
irisSetosaSepalWidth = statistics.mean(irisSetosa['Sepal_Width'])
irisVersicolorSepalWidth = statistics.mean(irisVersicolor['Sepal_Width'])
irisVirginicaSepalWidth = statistics.mean(irisVirginica['Sepal_Width'])

print('Average Sepal_Width Iris-setosa:', irisSetosaSepalWidth)
print('Average Sepal_Width Iris-versicolor:', irisVersicolorSepalWidth)
print('Average Sepal_Width Iris-virginica:', irisVirginicaSepalWidth)

print('\nb) which class has the smallest sepal width on average? Iris-versicolor')  # noqa


print(space)

# c) rank each class in order of homogeneity
print('\nc) rank each class in order of homogeneity')

# With Coefficient of Variation
irisSetosaHomogeneity = statistics.pstdev(irisSetosa['Sepal_Length'])/statistics.mean(irisSetosa['Sepal_Length'])  # noqa
irisVersicolorHomogeneity = statistics.pstdev(irisVersicolor['Sepal_Length'])/statistics.mean(irisVersicolor['Sepal_Length'])  # noqa
irisVirginicaHomogeneity = statistics.pstdev(irisVirginica['Sepal_Length'])/statistics.mean(irisVirginica['Sepal_Length'])  # noqa

print('Coefficient of variation Iris-setosa:', irisSetosaHomogeneity)
print('Coefficient of variation Iris-versicolor:', irisVersicolorHomogeneity)
print('Coefficient of variation Iris-virginica:', irisVirginicaHomogeneity)

print(space)
