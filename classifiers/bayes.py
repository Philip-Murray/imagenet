from model import *
import numpy as np

class BinaryNaiveBayes(Model):

    def __init__(self, inVectorSize):
        super().__init__(inVectorSize, 2)

        self.prob_xi0_y0 = np.ones(self.features)
        self.prob_xi1_y0 = np.zeros(self.features)
        self.prob_xi0_y1 = np.ones(self.features)
        self.prob_xi1_y1 = np.zeros(self.features)

        self.prob_y0 = 0
        self.prob_y1 = 1

    def predict(self, Xv):
        alpha_prob_y0 = self.prob_y0
        alpha_prob_y1 = self.prob_y1

        for f in range(self.features):
            if Xv[f] == 0:
                alpha_prob_y0 *= self.prob_xi0_y0 
                alpha_prob_y1 *= self.prob_xi0_y1
            else:
                alpha_prob_y0 *= self.prob_xi1_y0 
                alpha_prob_y1 *= self.prob_xi1_y1 

        if alpha_prob_y0 > alpha_prob_y1:
            return 0
        else:
            return 1        

    def fit(self, X_set, Y_set, epochs, report_progress=True, collect_accuracy=True):

        if X_set.shape[1] != self.features:
            print("ERROR - invalid dataset dims for model input")
            return

        cases_y0 = 0
        cases_y1 = 0

        cases_x0_y0 = np.zeros(self.features)
        cases_x1_y0 = np.zeros(self.features)
        cases_x0_y1 = np.zeros(self.features)
        cases_x1_y1 = np.zeros(self.features)

        for b in range(X_set.shape(0)):
            y_act = Y_set[b]
            
            if y_act == 0:
                cases_y0 += 1
            else:
                cases_y1 += 1
                
            for f in range(self.features):
                x_val = X_set[b, f]

                if x_val == 0 and y_act == 0:
                    cases_x0_y0[f] += 1
                if x_val == 1 and y_act == 0:
                    cases_x1_y0[f] += 1
                if x_val == 0 and y_act == 1:
                    cases_x0_y1[f] += 1
                if x_val == 1 and y_act == 1:
                    cases_x1_y1[f] +=1

        self.prob_xi0_y0 = cases_x0_y0 / cases_y0
        self.prob_xi1_y0  = cases_x1_y0 / cases_y0
        self.prob_xi0_y1 = cases_x0_y1 / cases_y1
        self.prob_xi1_y1 = cases_x1_y1 / cases_y1

        self.prob_y0 = cases_y0 / (cases_y0 + cases_y1)
        self.prob_y1 = cases_y1 / (cases_y0 + cases_y1)


    def accuracy_test(self, X_set, Y_set, report_progress=True):
        if X_set.shape[0] != self.features:
            print("ERROR - invalid dataset dims for model input")
            return
        if X_set.shape[0] != Y_set.shape[0]:
            print("ERROR - |X_set| != |Y_set|")
            return
        batch = X_set.shape[0]

        error_sum = 0
        for b in range(batch):
            pred = self.predict(X_set[b])
            if pred != Y_set[b]:
                error_sum += 1
        
        err = error_sum / batch
        return err

            


    def predict_batch(self, X_set, Y_set, report_progress=True):
        pass



        
