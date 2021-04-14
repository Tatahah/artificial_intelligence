#Kacper Tumulec 44535
#SI LAB6

import random
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self):
        self.void = None
        self.w = None
        self.n = None
        self.X_train = None

    def __correction_checker(self, result, expected):
        wrong_clas = []
        if np.array_equal(result, expected):
            return False
        for idx in range(len(result)):
            if result[idx] != expected[idx]:
                wrong_clas.append(idx)
        return wrong_clas

    def __stepfwrd(self, omega, input):
        wartosc = omega * input
        for idx, val in enumerate(wartosc):
            row_sum = np.sum(val)
            if row_sum > 0:
                self.void[idx] = 1
            else:
                self.void[idx] = 0
        return self.void

    def __omega_correction(self, omega, idx):
        if self.void[idx] == 0:
            for i in range(len(omega)):
                omega[i] = omega[i] + self.n * self.X_train[idx][i]
        else:
            for i in range(len(omega)):
                omega[i] = omega[i] - self.n * self.X_train[idx][i]

    def fit(self, X, y, lr, printiter = False):
        self.X_train = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))
        self.n = lr
        iterations = 1
        iterationstotal = 0
        rozmiar, _ = np.shape(self.X_train)
        self.void = np.zeros(rozmiar, dtype=int)
        self.w = np.array([0, 0, 0], dtype=float)
        self.void = self.__stepfwrd(self.w, self.X_train)
        while self.__correction_checker(self.void, y):
            idx_to_correct = self.__correction_checker(self.void, y)
            idx_cor = random.choice(idx_to_correct)
            self.__omega_correction(self.w, idx_cor)
            self.__stepfwrd(self.w, self.X_train)
            iterations += 1

        if printiter:
            print("Dopasowanie nastąpiło po " + str(iterations+iterationstotal) + " iteracjach")
        return iterations+iterationstotal, self.w


    def predict(self, X, plot=False):
        X_temp = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))
        rozmiar, _ = np.shape(X_temp)
        self.void = np.zeros(rozmiar, dtype=int)
        res = self.__stepfwrd(self.w, X_temp)
        print(res)
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.scatter(X[:, 0], X[:, 1], marker='x', c=res, s=25, edgecolor='k')
            x0_1 = np.amin(X_train[:, 0])
            x0_2 = np.amax(X_train[:, 0])
            x1_1 = (-self.w[0] * x0_1 - self.w[2]) / self.w[1]
            x1_2 = (-self.w[0] * x0_2 - self.w[2]) / self.w[1]
            ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')
            ymin = np.amin(X[:, 1])
            ymax = np.amax(X[:, 1])
            ax.set_ylim([ymin - 3, ymax + 3])
            plt.show()


n = 0.3
m = 50
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_classes=2, n_samples=m, class_sep=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

perc = Perceptron()
print(perc.fit(X_train, y_train, n, printiter=True))
perc.predict(X_test, plot=True)
print(y_test)