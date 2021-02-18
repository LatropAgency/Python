import pandas as pds
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris_dataset = load_iris()
print("Ключи iris_dataset: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Названияответов: {}".format(iris_dataset['target_names']))
print("Названияпризнаков: \n{}".format(iris_dataset['feature_names']))
print("Типмассива data: {}".format(type(iris_dataset['data'])))
print("Формамассива data: {}".format(iris_dataset['data'].shape))
print("Первыепятьстрокмассива data:\n{}".format(iris_dataset['data'][:5]))
print("Типмассива target: {}".format(type(iris_dataset['target'])))
print("Формамассива target: {}".format(iris_dataset['target'].shape))
print("Ответы:\n{}".format(iris_dataset['target']))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'],
    random_state=0)
print("формамассива X_train: {}".format(X_train.shape))
print("формамассива y_train: {}".format(y_train.shape))
print("формамассива X_test: {}".format(X_test.shape))
print("формамассива y_test: {}".format(y_test.shape))
iris_dataframe = pds.DataFrame(X_train, columns=iris_dataset.feature_names)
from pandas.plotting import scatter_matrix

grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                     hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()
