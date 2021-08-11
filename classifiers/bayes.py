from model import *

class BinaryNaiveBayes(Model):

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