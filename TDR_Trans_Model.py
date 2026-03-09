# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 19:21:18 2023

@author: Zhuangji.Wang
"""

from TDR_Trans_Modules import *

class TDR_Transformer(tf.keras.Model):
    
    def __init__(self, *, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           dropout_rate=dropout_rate)
        
        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           dropout_rate=dropout_rate)
        
        self.final_layer_ref = tf.keras.layers.Dense(units=2, activation='softmax')
        self.final_layer_er = tf.keras.layers.Dense(units=1, activation='linear')
        self.final_layer_mul = tf.keras.layers.Multiply()
        
    def call(self, inputs):
        
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        x  = inputs
        context = inputs
    
        context = self.encoder(context)  # (batch_size, context_len, d_model)
    
        x = self.decoder(x, context)  # (batch_size, target_len, d_model)
        
        # Final linear layer output.
        logits = self.final_layer_ref(x)  # (batch_size, target_len, target_vocab_size)
        # Final regression input.
#        er_reg = self.final_layer_er(x)
        # Final output after masking
        
#        return self.final_layer_mul([logits, er_reg])
        return logits
    
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        
    
        
def masked_loss(label, pred):
    
#    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction='none')
    loss = loss_object(label, pred)
    
    # mask = tf.cast(mask, dtype=loss.dtype)
    # loss *= mask
    
    loss = tf.reduce_sum(loss)
    return loss

          
        