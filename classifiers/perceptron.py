from model import *
import numpy as np

class Model:
    def __init__(self, inVectorSize: int, outVectorSize: int):
        self.features = inVectorSize
        self.classifications = outVectorSize

    def assert_inputdim(self, f: int):
        if self.features == f:
            return
        print("Model input length is "+str(self.features)+", given dataset with "+str(f))


    def fit(self, X_set, Y_set, epochs, report_progress=True):
        pass

    def accuracy_test(self, X_set, Y_set, report_progress=True):
        for index in range(Y_set.shape[0]):
            pass


    def predict_batch(self, X_set, Y_set, report_progress=True):
        pass

    def predict(self, X):
        pass



class BinaryPerceptron(Model):

    def __init__(self, inVectorSize):
        super().__init__(inVectorSize, 2)

        self.weight_v = 2*np.random.rand(inVectorSize) - 1
        self.bias = 0

    def predict(self, Xv):
        y = np.sum(self.weight_v * Xv) + self.bias
        if y < 0:
            return 0
        else:
            return 1

    def fit(self, X_set, Y_set, epochs, report_progress=True, collect_accuracy=True):
        lrn = 0.01
        for epoch in range(epochs):
            for b in range(X_set.shape(0)):

                pred = self.predict(X_set[b])

                if pred == Y_set[b]:
                    continue

                if pred == 0: #pred = 0 but y_act = 1
                    self.bias += lrn
                    self.weight_v = self.weight_v + lrn * X_set[b]
                else:
                    self.bias -= 0.01
                    self.weight_v = self.weight_v - lrn * X_set[b]



    def accuracy_test(self, X_set, Y_set, report_progress=True):
        error_sum = 0
        for b in range(Y_set.shape[0]):
            pred = self.predict(X_set[b])
            if pred != Y_set[b]:
                error_sum += 1
        return error_sum / Y_set.shape[0]
        
    def predict_batch(self, X_set, Y_set, report_progress=True):
        preds = []
        for b in range(X_set.shape(0)):
            preds.append(self.predict(X_set[b]))

        return np.array(preds)





class MultiPerceptron(Model):

    def __init__(self, inVectorSize, outVectorSize):
        super().__init__(inVectorSize, outVectorSize)

        self.weight_m = 2*np.random.rand(outVectorSize, inVectorSize) - 1
        self.bias_v = np.zeros(outVectorSize)

    def predict(self, Xv):
        Yv = np.matmul(self.weight_m, Xv) + self.bias_v
        return np.argmax(Yv)

    def fit(self, X_set, Y_set, epochs, report_progress=True, collect_accuracy=True):
        lrn = 0.01
        for epoch in range(epochs):
            for b in range(X_set.shape(0)):

                pred = self.predict(X_set[b])

                if pred == Y_set[b]:
                    continue

                self.bias_v[pred] -= lrn
                self.weight_m[pred] -= lrn * X_set[b]

                self.bias_v[Y_set[b]] += lrn
                self.weight_m[Y_set[b]] += lrn * X_set[b]



    def accuracy_test(self, X_set, Y_set, report_progress=True):
        error_sum = 0
        for b in range(Y_set.shape[0]):
            pred = self.predict(X_set[b])
            if pred != Y_set[b]:
                error_sum += 1
        return error_sum / Y_set.shape[0]
        
    def predict_batch(self, X_set, Y_set, report_progress=True):
        preds = []
        for b in range(X_set.shape(0)):
            preds.append(self.predict(X_set[b]))

        return np.array(preds)

    


    