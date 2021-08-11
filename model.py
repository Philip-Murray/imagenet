
import numpy as np
def ErrorFunction():
    pass
    

class Model:
    def __init__(self, inVectorSize: int, outVectorSize: int):
        self.features = inVectorSize
        self.classifications = outVectorSize

    def predict_batch(self, X_set, Y_set, report_progress=True):
        preds = []
        for b in range(X_set.shape(0)):
            pred = self.predict(X_set[b])
            preds.append(pred)

        return np.array(preds)

    def assert_inputdim(self, f: int):
        if self.features == f:
            return
        print("Model input length is "+str(self.features)+", given dataset with "+str(f))


    def fit(self, X_set, Y_set, epochs, report_progress=True, collect_accuracy=True,):
        pass

    def accuracy_test(self, X_set, Y_set, report_progress=True):
        for index in range(Y_set.shape[0]):
            pass


    def predict(X):
        pass

    






# Print iterations progress
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def go():
    import time

    # A List of Items
    items = list(range(0, 57))
    l = len(items)

    # Initial call to print 0% progress
    printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for i, item in enumerate(items):
        # Do stuff...
        time.sleep(0.1)
        # Update Progress Bar
        printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)