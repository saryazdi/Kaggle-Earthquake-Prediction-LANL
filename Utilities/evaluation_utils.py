# -*- coding: utf-8 -*-
'''
Copyright(c) 2019, Soroush Saryazdi
All rights reserved.
2019/03/26
'''
import numpy as np
import pandas as pd

def save_submission(dictionary, name):
    df = pd.DataFrame(list(dictionary.items()), columns=['seg_id', 'time_to_failure'])
    df = df.sort_values(by=['seg_id'])
    df.to_csv(f'submissions/{name}.csv', index=False)