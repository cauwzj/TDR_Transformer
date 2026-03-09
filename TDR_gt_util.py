# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:38:14 2023

@author: Zhuangji.Wang
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn

from tensorflow import einsum
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange

class PreNorm(Layer):
    def __init__(self, fn):
        super(PreNorm,self).__init__()
        self.norm=nn.LayerNormalization()
        self.fn=fn
    def call(self, x, training=True):
        return self.fn(self.norm(x),training=training)
    
class PosiEncoder(Layer):
    def __init__(self):
        super(PosiEncoder, self).__init__()
        pe=tf.zeros([50, 32], dtype=tf.float32)
        div_term=tf.exp(-tf.range(0,32,2, dtype=tf.float32)*(tf.math.log(10000.0)/32))
        
        for ii in tf.range(0,50,dtype=tf.float32):
            update_1=tf.concat([tf.expand_dims(tf.math.sin((ii+1.0)*div_term),1),tf.expand_dims(tf.math.cos((ii+1.0)*div_term),1)],axis=1)
            update_1=tf.reshape(update_1,[1,32])
            pe=tf.tensor_scatter_nd_update(pe,[[ii.numpy().astype(np.int32)]],update_1)
            
        self.pe=tf.expand_dims(pe, axis=0) 
        
    def call(self, x):
        x=x+tf.Variable(self.pe, trainable=False)
        return x
        
        
class CovEmbedding(Layer):
    def __init__(self):
        super(CovEmbedding,self).__init__()
                
        self.net=Sequential([
            nn.InputLayer(input_shape=(1000,1)),
            nn.Conv1D(filters=32, kernel_size=20, strides=20, padding='same')])
        
    def call(self, x, training=True):
        return self.net(x, training=training)
    
class LinearForward(Layer):
    def __init__(self):
        super(LinearForward,self).__init__()
        
        def gelu(x, approximate=False):

            if approximate:
                coeff = tf.cast(0.044715, x.dtype)
                return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
            else:
                return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))
        
        self.net=Sequential([
            nn.Dense(64),
            nn.Activation(gelu),
            nn.Dense(32)
        ])
        
    def call(self, x, training=True):
        return self.net(x, training=training)
    
class Attention(Layer):
    def __init__(self):
        super(Attention, self).__init__()
        inner_dim=32
        self.heads=4
        self.scale=8**-0.5
        self.attend=nn.Softmax()
        self.to_qkv=nn.Dense(inner_dim*3, use_bias=False)
        self.to_out=nn.Dense(inner_dim)
        
    def call(self, x, training=True):
        qkv=self.to_qkv(x)
        qkv=tf.split(qkv, num_or_size_splits=3, axis=-1)
        q,k,v=map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots=einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn=self.attend(dots)
        
        x=einsum('b h i j, b h j d-> b h i d', attn, v)
        x=rearrange(x, 'b h n d -> b n (h d)')
        x=self.to_out(x, training=training)
        
        return x
    
class MasedAttention(Layer):
    def __init__(self):
        super(MasedAttention, self).__init__()
        inner_dim=32
        self.heads=4
        self.scale=8**-0.5
        self.attend=nn.Softmax()
        self.to_qkv=nn.Dense(inner_dim*3, use_bias=False)
        self.to_out=nn.Dense(inner_dim)
        
        attn_shape=(1,50,50)
        subsequent_mask=np.triu(np.ones(attn_shape),k=1).astype(np.int8)
        self.mask=tf.convert_to_tensor(1-subsequent_mask)
        
        
    def call(self, x, training=True):
        qkv=self.to_qkv(x)
        qkv=tf.split(qkv, num_or_size_splits=3, axis=-1)
        q,k,v=map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots=einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        mask_active=tf.ones_like(dots,dtype=tf.int8)*self.mask
        dots=tf.where(mask_active==1,dots,-1e6) 
        attn=self.attend(dots)
        
        x=einsum('b h i j, b h j d-> b h i d', attn, v)
        x=rearrange(x, 'b h n d -> b n (h d)')
        x=self.to_out(x, training=training)
        
        return x
    
class MemoryAttention(Layer):
    def __init__(self):
        super(MemoryAttention, self).__init__()
        inner_dim=32
        self.heads=4
        self.scale=8**-0.5
        self.attend=nn.Softmax()
        self.to_qk=nn.Dense(inner_dim*2, use_bias=False)
        self.to_v=nn.Dense(inner_dim, use_bias=False)
        self.to_out=nn.Dense(inner_dim)
        
        
    def call(self, x, m, training=True):
        qk=self.to_qk(m)
        v=self.to_v(x)
        qk=tf.split(qk, num_or_size_splits=2, axis=-1)
        q,k=map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qk)
        v=rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        dots=einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn=self.attend(dots)
        
        x=einsum('b h i j, b h j d-> b h i d', attn, v)
        x=rearrange(x, 'b h n d -> b n (h d)')
        x=self.to_out(x, training=training)
        
        return x
    
class EncoderStructure(Layer):
    def __init__(self):
        super(EncoderStructure,self).__init__()
        self.layers=[]
        for _ in range(2):
            self.layers.append([
                PreNorm(Attention()),
                PreNorm(LinearForward())
            ])
            
    def call(self, x, training=True):
        for attn, LinearForward in self.layers:
            x=attn(x, training=training)+x
            x=LinearForward(x, training=training)+x
            
        return x
    
class DecoderStructure(Layer):
    def __init__(self):
        super(DecoderStructure,self).__init__()
        self.layers=[]
        for _ in range(2):
            self.layers.append([
                PreNorm(MasedAttention()),
                PreNorm(MemoryAttention()),
                PreNorm(LinearForward())
            ])
            
    def call(self, x, m, training=True):
        for maskedattn, memoryattn, linearffd in self.layers:
            x=maskedattn(x, training=training)+x
            x=memoryattn(x, m, training=training)+x
            x=linearffd(x, training=training)+x
            
        return x
    
class TDR_GT(Model):
    def __init__(self):
        super(TDR_GT, self).__init__()
        self.COvEmbed=CovEmbedding()
        self.PosiEmbed=PosiEncoder()
        self.encode=EncoderStructure()
        self.decode=DecoderStructure()
    
    
            
        
        
        
        
        