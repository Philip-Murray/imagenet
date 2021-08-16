import loader, features, persistence, model, timeit, os
from algorithms.ann import *
from algorithms.perceptron import *
from algorithms.bayes import *
import numpy as np
import pandas as pd
A4_PATH = os.path.dirname(os.path.realpath(__file__))

TRIAL_COUNT = 5
MODEL_COUNT = 3

FACES, MNIST = 0, 1
PERCEPT, BAYES, ANN = 0, 1, 2

reflect_dataset = {FACES: "faces", MNIST: "mnist"}
reflect_ary     = {FACES: "binary", MNIST: "multiclass"}
reflect_classifier = {PERCEPT: "PERCEPTRON", BAYES: "NAIVE BAYES", ANN: "NEURAL NETWORK"}



class ExperimentContainer: #Class to do problem 3

    def __init__(self):
        self.tests = {}
        for percent in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            self.tests[percent] = {}
            for t in range(TRIAL_COUNT):
                self.tests[percent][t] = persistence.NewSession()

    def get_model(self, dataset, percent, classifier, trial):
        return self.tests[percent][trial].get_model(dataset, classifier)


    def run(self, report_progress=True, save_df=True, epoch_set=1):
        def get_percentof_dataset(dataset, percent):
            rg = int((percent / 100) * dataset.shape[0])
            return dataset[:rg]

        def getDataset(id):
            if id == 1:
                return features.mnist
            else:
                return features.faces
        def announce(rp, d):
            if d == FACES and rp:
                print()
                print()
                print("FACES Dataset ---")
                print()
            if d == MNIST and rp:
                print()
                print()
                print("MNIST Dataset ---")
                print()

            
        def report(dataset, classifier, percent, accuracies, runtime):
            avg = sum(accuracies) / len(accuracies)
            sdv = np.std(accuracies)

            avg_t = sum(runtime) / len(runtime)
            sdv_t = np.std(runtime)

            ra = reflect_ary[dataset]
            rc = reflect_classifier[classifier]
            rd = reflect_dataset[dataset]
            print(str(percent)+chr(37)+" of "+rd+" - "+ra+" "+rc+":")
            print("train_time="+str(round(avg_t*1000, 2))+"ms, time_std="+str(round(sdv_t*100, 2))+"  avg_accuracy="+str(round(avg*100, 2))+chr(37)+" std="+str(round(sdv*100, 2)))
            print()




        df_tuples = []

        for DATASET in [FACES, MNIST]:
            announce(report_progress, DATASET)
            for percent in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:

                X_train = get_percentof_dataset(getDataset(DATASET).X_train, percent)
                Y_train = get_percentof_dataset(getDataset(DATASET).Y_train, percent)

                X_test = get_percentof_dataset(getDataset(DATASET).X_test, percent)
                Y_test = get_percentof_dataset(getDataset(DATASET).Y_test, percent)

                for classifier in [PERCEPT, BAYES, ANN]:  
                    accs = [] 
                    ts = []

                    for trial in range(TRIAL_COUNT):
                        m = self.get_model(DATASET, percent, classifier, trial)

                        start = timeit.default_timer()

                        m.fit(X_train, Y_train, epochs=epoch_set)
                        runtime = timeit.default_timer()  - start

                        accuracy = m.accuracy_test(X_test, Y_test)

                        accs.append(accuracy)
                        ts.append(runtime)
                        df_tuples.append([DATASET, classifier, percent, trial, 100*accuracy, 1000*runtime])
                    

                    if report_progress:
                        report(DATASET, classifier, percent, accs, ts)
                if report_progress:
                    print()
                    print()
        if save_df:
            df = pd.DataFrame(df_tuples, columns = ['Dataset', 'Model', 'Percent', 'Trial',     'Accuracy', 'Time'])
            df.to_csv(A4_PATH+'/metrics.csv', index=False)
        return df









#Copying the above class, and removing stuff so that it only does a subcase

class TrainGlobalSessionModels:

    def __init__(self, percent, global_session_ref):
        self.percent = percent
        self.global_session_ref = global_session_ref

    def get_model(self, dataset, percent, classifier, trial):
        return self.global_session_ref.get_model(dataset, classifier)


    def run(self, report_progress=True, save_df=True, epoch_set=1):
        def get_percentof_dataset(dataset, percent):
            rg = int((percent / 100) * dataset.shape[0])
            return dataset[:rg]

        def getDataset(id):
            if id == 1:
                return features.mnist
            else:
                return features.faces
        def announce(rp, d):
            if d == FACES and rp:
                print()
                print()
                print("FACES Dataset ---")
                print()
            if d == MNIST and rp:
                print()
                print()
                print("MNIST Dataset ---")
                print()

            
        def report(dataset, classifier, percent, accuracies, runtime):
            avg = sum(accuracies) / len(accuracies)
            sdv = np.std(accuracies)

            avg_t = sum(runtime) / len(runtime)
            sdv_t = np.std(runtime)

            ra = reflect_ary[dataset]
            rc = reflect_classifier[classifier]
            rd = reflect_dataset[dataset]
            print(str(percent)+chr(37)+" of "+rd+" - "+ra+" "+rc+":")
            print("train_time="+str(round(avg_t*1000, 2))+"ms,  accuracy="+str(round(avg*100, 2))+chr(37)+" std="+str(round(sdv*100, 2)))
            print()




        df_tuples = []

        for DATASET in [FACES, MNIST]:
            announce(report_progress, DATASET)
            for percent in [self.percent]:

                X_train = get_percentof_dataset(getDataset(DATASET).X_train, percent)
                Y_train = get_percentof_dataset(getDataset(DATASET).Y_train, percent)

                X_test = get_percentof_dataset(getDataset(DATASET).X_test, percent)
                Y_test = get_percentof_dataset(getDataset(DATASET).Y_test, percent)

                for classifier in [PERCEPT, BAYES, ANN]:  
                    accs = [] 
                    ts = []

                    for trial in [0]:
                        m = self.get_model(DATASET, percent, classifier, trial)

                        start = timeit.default_timer()

                        m.fit(X_train, Y_train, epochs=epoch_set)
                        runtime = timeit.default_timer()  - start

                        accuracy = m.accuracy_test(X_test, Y_test)

                        accs.append(accuracy)
                        ts.append(runtime)

                    if report_progress:
                        report(DATASET, classifier, percent, accs, ts)
                if report_progress:
                    print()
                    print()



def TestGlobalModel(sess, ds, mt, f, l):
    def getDataset(id):
        if id == 1:
            return features.mnist
        else:
            return features.faces

    xt = getDataset(ds).X_test[f:l]
    yt = getDataset(ds).Y_test[f:l]

    mdl = sess.get_model(ds, mt)

    ar = mdl.predict_batch(xt, yt)

    for i in range(f, l):
        z = ar[i-f]
        cr = "[INCORRECT]"
        if yt[i-f] == z:
            cr = "[CORRECT]"
        print(reflect_dataset[ds]+" test image "+str(i)+": Actual: "+str(int(yt[i-f]))+" Predicted: "+str(int(z))+" "+cr)

    




