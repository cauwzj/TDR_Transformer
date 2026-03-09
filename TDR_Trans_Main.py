# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 19:05:26 2023

@author: Zhuangji.Wang
"""

import pandas as pd
import numpy as np

from TDR_Trans_Modules import *
from TDR_Trans_Model import *

WaveForm=pd.read_csv('Er_0_WaveForm.csv', header=None).astype(np.float32)
WaveForm=WaveForm.to_numpy(copy=True)

WaveInfo=pd.read_csv('Er_0_Info.csv', header=None).astype(np.uint8)
WaveInfo=WaveInfo.to_numpy(copy=True)
WaveInfo=tf.expand_dims(WaveInfo, axis=2)

# x=WaveForm
# y=WaveForm

# encoder = Encoder(num_layers=4, d_model=32, num_heads=4, dff=128, dropout_rate=0.1)
# x=encoder(x)
# decoder = Decoder(num_layers=4, d_model=32, num_heads=4, dff=128, dropout_rate=0.1)
# y=decoder(y,x)

d_model=32

transformer = TDR_Transformer(
    num_layers=2,
    d_model=d_model,
    num_heads=4,
    dff=64,
    dropout_rate=0.1)
res=transformer(WaveForm)
transformer.summary()

TDR_learning_rate = CustomSchedule(d_model)
TDR_optimizer = tf.keras.optimizers.Adam(TDR_learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

transformer.compile(optimizer=TDR_optimizer, loss=masked_loss)

# transformer.compile(optimizer=TDR_optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

history_reg=transformer.fit(WaveForm, WaveInfo, epochs=100)