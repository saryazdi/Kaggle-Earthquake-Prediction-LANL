# -*- coding: utf-8 -*-
'''
Copyright(c) 2019, Soroush Saryazdi
All rights reserved.
2019/03/26
'''
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output

def save_submission(dictionary, name):
    df = pd.DataFrame(list(dictionary.items()), columns=['seg_id', 'time_to_failure'])
    df = df.sort_values(by=['seg_id'])
    df.to_csv(f'submissions/{name}.csv', index=False)


class LossPlot(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="Training")
        plt.plot(self.x, self.val_losses, label="Validation")
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.grid()
        plt.legend()
        plt.show();