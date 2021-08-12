from model import *
import numpy as np

class BinaryPerceptron(Model):

    def save_to_session(self):
        pass#g_params.session.face_p = self

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
            err_count = 0
            for b in range(X_set.shape[0]):

                pred = self.predict(X_set[b])

                if pred == Y_set[b]:
                    continue
                
                err_count += 1
                if pred == 0: #pred = 0 but y_act = 1
                    self.bias += lrn
                    self.weight_v = self.weight_v + lrn * X_set[b]
                else:
                    self.bias -= lrn
                    self.weight_v = self.weight_v - lrn * X_set[b]

            print("Bi_Percpt - Epoch "+str(epoch)+", accuracy = "+str(1-(err_count / X_set.shape[0]))+" ")




class MultiClassPerceptron(Model):

    def save_to_session(self):
        pass#g_params.session.mnsit_p = self

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
            err_count = 0
            for b in range(X_set.shape[0]):

                pred = self.predict(X_set[b])

                if pred == Y_set[b]:
                    continue

                err_count += 1
                self.bias_v[pred] -= lrn
                self.weight_m[pred] -= lrn * X_set[b]

                self.bias_v[Y_set[b]] += lrn
                self.weight_m[Y_set[b]] += lrn * X_set[b]
            print("MC_Percpt - Epoch "+str(epoch)+", accuracy = "+str(1-(err_count / X_set.shape[0]))+" ")
            

    


    