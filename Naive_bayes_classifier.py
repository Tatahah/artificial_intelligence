# Sztuczna inteligencja LAB5
# Kacper Tumulec 44535

import numpy as np
import math

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split


class BayesDysk:
    bins_used = 0
    laplace = False
    post_res = []
    prio_res = []

    def __init__(self, laplace=False):
        self.laplace = laplace

    def fit(self, X, y):
        testU = np.unique(y)
        bu = np.unique(X).shape[0]
        self.bins_used = bu
        classescount = testU.shape
        prio = np.zeros(classescount, dtype=np.float64)
        samplecount, featurecount = X.shape
        results = []
        for idx, c in enumerate(testU):
            testx = X[c == y]
            resclass = []
            dim1, dim2 = testx.shape
            if self.laplace:
                prio[idx] = (testx.shape[0] + 1) / (float(samplecount) + self.bins_used)
            else:
                prio[idx] = testx.shape[0] / float(samplecount)

            for j in range(featurecount):
                resfeature = []
                for i in range(self.bins_used):
                    if self.laplace:
                        resfeature.append((np.count_nonzero(testx[:, j] == i) + 1) / (dim1 + self.bins_used))
                    else:
                        resfeature.append(np.count_nonzero(testx[:, j] == i) / dim1)
                resclass.append(resfeature)
            results.append(resclass)
        self.post_res = results
        self.prio_res = prio

    def predict(self, data_to_clasify):
        predRes = []
        for inpu in data_to_clasify:
            predResTemp = []
            for cidx, predi in enumerate(self.post_res):
                tempinteger = predi[0][int(inpu[0])]
                for classRan in range(1, len(inpu)):
                    tempinteger = tempinteger * predi[classRan][int(inpu[classRan])]
                tempinteger = tempinteger * self.prio_res[cidx]
                predResTemp.append(tempinteger)
            predRes.append(predResTemp)

        finalowewyiki = []
        for finalowe in range(len(predRes)):
            finalowewyiki.append(np.argmax(predRes[finalowe]) + 1)
        return finalowewyiki


class BayesCiag:
    bmean = []
    stdev = []

    def fit(self, X, y):
        testU = np.unique(y)
        samplecount, featurecount = X.shape
        classescount = len(testU)
        tempmean = np.zeros((classescount, featurecount), dtype=np.float64)
        for idx, c in enumerate(testU):
            testx = X[c == y]
            tempdev = []
            tempmean[idx, :] = testx.mean(axis=0)
            for j in range(featurecount):
                tempsum = 0
                for i in range(len(testx)):
                    tempsum = tempsum + math.pow(testx[i][j] - tempmean[idx][j], 2)
                tempdev.append(math.sqrt((1 / (len(testx) - 1)) * tempsum))
            self.stdev.append(tempdev)
        self.bmean = tempmean

    def predict(self, X):
        predRes = []
        for inpu in X:
            predResTemp = []
            for cidx, predi in enumerate(self.stdev):
                nexttemp = 1
                for vidx, value in enumerate(predi):
                    nexttemp = nexttemp * (1 / (math.sqrt(2 * math.pi) * value)) * math.exp(
                        -(math.pow((inpu[vidx] - self.bmean[cidx][vidx]), 2)) / (2 * math.pow(value, 2)))
                predResTemp.append(nexttemp)
            predRes.append(np.argmax(predResTemp) + 1)
        return predRes


def accuracy(reslist, testlist):
    correct = 0
    for idx, val in enumerate(reslist):
        if val == testlist[idx]:
            correct = correct + 1
    return "{:.2f}".format(correct/len(reslist)*100) + '%'


wines = np.genfromtxt('wine.data', delimiter=",")
X = wines[:, 1:]
y = wines[:, 0]
features = 7
disc = KBinsDiscretizer(n_bins=features, encode='ordinal', strategy='uniform')
X = disc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X2 = wines[:, 1:]
y2 = wines[:, 0]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.33, random_state=42)


bayesdysk = BayesDysk(laplace=False)
bayesdysk.fit(X_train, y_train)
wyniki_dysk = bayesdysk.predict(X_test)
print("Dokladnosc klasyfikatora dyskretnego bez poprawki LaPlace'a: " + accuracy(wyniki_dysk, y_test))

bayesdysklp = BayesDysk(laplace=True)
bayesdysklp.fit(X_train, y_train)
wyniki_dysklp = bayesdysklp.predict(X_test)
print("Dokladnosc klasyfikatora dyskretnego z poprawka LaPlace'a: " + accuracy(wyniki_dysklp, y_test))

bayesciag = BayesCiag()
bayesciag.fit(X_train2, y_train2)
wyniki_ciag = bayesciag.predict(X_test2)
print("Dokladnosc klasyfikatora ciaglego: " + accuracy(wyniki_ciag, y_test2))

gnb = GaussianNB()
y_pred = gnb.fit(X_train2, y_train2).predict(X_test2)
print("Dokladnosc klasyfikatora GaussianNB z biblioteki sklearn: " + accuracy(y_pred, y_test2))
