# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 18:45:11 2023

@author: Zhuangji.Wang
"""

import pandas as pd

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import Model
# from tensorflow.keras.layers import Layer
# from tensorflow.keras import Sequential
# import tensorflow.keras.layers as nn

# from tensorflow import einsum
# from einops import rearrange, repeat
# from einops.layers.tensorflow import Rearrange

from TDR_gt_util import *


WaveForm=pd.read_csv('Er_0_WaveForm.csv', header=None).astype(np.float32)
WaveForm=WaveForm.to_numpy(copy=True)

# TDR_MLP=MLP()
# TDR_COV=CovEmbedding()
# TDR_Posi=PosiEncoder()

# output=TDR_Posi(TDR_COV(WaveForm))
# TDR_Norm=PreNorm(TDR_Posi)

# TDR_Encoder=EncoderStructure()
# output=TDR_Encoder(output)

main_net=Sequential([CovEmbedding(),
                     PosiEncoder(),
                     EncoderStructure()])

output=main_net(WaveForm)

