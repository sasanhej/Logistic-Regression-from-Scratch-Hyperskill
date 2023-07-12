import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import math
class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    def predict_proba(self, row, coef_):
        if self.fit_intercept:
            t = coef_[0] + np.dot(row, coef_[1:])
        else:
            t = np.dot(row, coef_)
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        k = 0
        self.coef_ = [0. for _ in range(X_train.shape[1])]
        if self.fit_intercept:
            self.coef_ = [0.] + self.coef_
            k = 1
        self.epoch = []
        for l in range(self.n_epoch):
            errors = []
            for i, row in enumerate(X_train):
                y_hat = self.predict_proba(row, self.coef_)
                delta = self.l_rate * (y_hat - y_train[i]) * y_hat * (1 - y_hat)
                errors.append(((y_hat - y_train[i])**2) * 1/len(y_train))
                self.coef_[0] = self.coef_[0] - delta
                # print('delta = ', delta)
                for j in range(len(row)):
                    self.coef_[j + k] = self.coef_[j + k] - delta * row[j]
            self.epoch.append(errors)

    def fit_log_loss(self, X_train, y_train):
        k = 0
        self.coef_ = [0. for _ in range(X_train.shape[1])]
        if self.fit_intercept:
            self.coef_ = [0.] + self.coef_
            k = 1
        self.epoch = []
        for l in range(self.n_epoch):
            errors = []
            for i, row in enumerate(X_train):
                y_hat = self.predict_proba(row, self.coef_)
                delta = self.l_rate * (y_hat - y_train[i]) / len(y_train)
                errors.append((-(y_train[i] * math.log(y_hat) + (1 - y_train[i]) * math.log(1 - y_hat))) * 1/len(y_train))
                self.coef_[0] = self.coef_[0] - delta
                # print('delta = ', delta)
                for j in range(len(row)):
                    self.coef_[j + k] = self.coef_[j + k] - delta * row[j]
            self.epoch.append(errors)

    def predict(self, X_test, cut_off=0.5):
        predictions = []
        for row in X_test:
            y_hat = self.predict_proba(row, self.coef_)
            predictions.append(1 if y_hat >= cut_off else 0)
        return np.array(predictions)


from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(as_frame=True)
df = data.frame
X = df[['worst concave points', 'worst perimeter', 'worst radius']]
y = df['target']

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

clr1 = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
clr1.fit_mse(X_train, y_train.to_numpy())
y_pred1 = clr1.predict(X_test)
mse_accuracy = accuracy_score(y_test, y_pred1)
error1 = clr1.epoch

clr2 = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
clr2.fit_log_loss(X_train, y_train.to_numpy())
y_pred2 = clr2.predict(X_test)
logloss_accuracy = accuracy_score(y_test, y_pred2)
error2 = clr2.epoch

clr3 = LogisticRegression().fit(X_train, y_train.to_numpy())
y_pred3 = clr3.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, y_pred3)

answer_dict = {'mse_accuracy': mse_accuracy,
               'logloss_accuracy': logloss_accuracy,
               'sklearn_accuracy': sklearn_accuracy,
               'mse_error_first': error1[0],
               'mse_error_last': error1[-1],
               'logloss_error_first': error2[0],
               'logloss_error_last': error2[-1]}
print(answer_dict)

print(f"""
Answers to the questions:
1) {min(error1[0]):.5f}
2) {min(error1[-1]):.5f}
3) {max(error2[0]):.5f}
4) {max(error2[-1]):.5f}
5) expanded
6) expanded
""")

