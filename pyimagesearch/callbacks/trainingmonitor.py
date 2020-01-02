from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

    def on_train_begin(self, logs={}):
        # initialize the history
        self.H = {}
        # if jsonPath is not none load training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())
                for k in self.H.keys():
                    self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        for (k, v) in logs:
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l

        if self.jsonPath is not None:
            with open(jsonPath, 'w') as file_:
                file_.write(json.dumps(self.H))

        if len(self.H['loss']) > 1:
            N = np.arange(0, len(self.H['loss']))

            plt.style.use('ggplot')
            plt.figure()
            plt.plot(N, self.H['loss'], label='train_loss')
            plt.plot(N, self.H['val_loss'], label='val_loss')
            plt.plot(N, self.H['acc'], label='train_acc')
            plt.plot(N, self.H['val_acc'], label='val_acc')
            plt.xlabel("# Epochs")
            plt.ylabel(
                "Loss / Accuracy [Epoch {}]".format(len(self.H['loss'])))
            plt.legend()
            plt.savefig(self.figPath)
            plt.close()
