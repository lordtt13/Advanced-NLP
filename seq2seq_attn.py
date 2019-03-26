# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 00:08:21 2019

@author: tanma
"""

import os, sys

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.layers import  Bidirectional, RepeatVector, Concatenate, Activation, Dot, Lambda
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K

import numpy as np
import matplotlib.pyplot as plt

if len(K.tensorflow_backend._get_available_gpus()) > 0:
  from keras.layers import CuDNNLSTM as LSTM
  from keras.layers import CuDNNGRU as GRU

def softmax_over_time(x):
  assert(K.ndim(x) > 2)
  e = K.exp(x - K.max(x, axis=1, keepdims=True))
  s = K.sum(e, axis=1, keepdims=True)
  return e / s

BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 256
LATENT_DIM_DECODER = 256 
NUM_SAMPLES = 10000
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

input_texts = [] 
target_texts = []
target_texts_inputs = []

t = 0
with open('hin.txt','rb') as f:
    lines = [x.decode('utf8').strip() for x in f.readlines()]
    
for line in lines:
  t += 1
  if t > NUM_SAMPLES:
    break

  if '\t' not in line:
    continue

  input_text, translation = line.rstrip().split('\t')

  target_text = translation + ' <eos>'
  target_text_input = '<sos> ' + translation

  input_texts.append(input_text)
  target_texts.append(target_text)
  target_texts_inputs.append(target_text_input)


tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

word2idx_inputs = tokenizer_inputs.word_index

max_len_input = max(len(s) for s in input_sequences)

tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs) 
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)

word2idx_outputs = tokenizer_outputs.word_index

num_words_output = len(word2idx_outputs) + 1

max_len_target = max(len(s) for s in target_sequences)

encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)

decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')

decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')

word2vec = {}
with open(os.path.join('E:\\Misc\\Glove Data/glove.6B.%sd.txt' % EMBEDDING_DIM),'rb') as f:
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec

num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx_inputs.items():
  if i < MAX_NUM_WORDS:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector
      
embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=max_len_input,
  # trainable=True
)

decoder_targets_one_hot = np.zeros(
  (
    len(input_texts),
    max_len_target,
    num_words_output
  ),
  dtype='float32'
)

for i, d in enumerate(decoder_targets):
  for t, word in enumerate(d):
    decoder_targets_one_hot[i, t, word] = 1
    
encoder_inputs_placeholder = Input(shape=(max_len_input,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = Bidirectional(LSTM(
  LATENT_DIM,
  return_sequences=True,
  # dropout=0.5 
))
encoder_outputs = encoder(x)

decoder_inputs_placeholder = Input(shape=(max_len_target,))

decoder_embedding = Embedding(num_words_output, EMBEDDING_DIM)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

attn_repeat_layer = RepeatVector(max_len_input)
attn_concat_layer = Concatenate(axis=-1)
attn_dense1 = Dense(10, activation='tanh')
attn_dense2 = Dense(1, activation=softmax_over_time)
attn_dot = Dot(axes=1) 

def one_step_attention(h, st_1):
  st_1 = attn_repeat_layer(st_1)

  x = attn_concat_layer([h, st_1])

  x = attn_dense1(x)

  alphas = attn_dense2(x)

  context = attn_dot([alphas, h])

  return context

decoder_lstm = LSTM(LATENT_DIM_DECODER, return_state=True)
decoder_dense = Dense(num_words_output, activation='softmax')

initial_s = Input(shape=(LATENT_DIM_DECODER,), name='s0')
initial_c = Input(shape=(LATENT_DIM_DECODER,), name='c0')
context_last_word_concat_layer = Concatenate(axis=2)

s = initial_s
c = initial_c

outputs = []
for t in range(max_len_target): 
  context = one_step_attention(encoder_outputs, s)

  selector = Lambda(lambda x: x[:, t:t+1])
  xt = selector(decoder_inputs_x)
  
  decoder_lstm_input = context_last_word_concat_layer([context, xt])

  o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[s, c])

  decoder_outputs = decoder_dense(o)
  outputs.append(decoder_outputs)
  
def stack_and_transpose(x):
  x = K.stack(x) 
  x = K.permute_dimensions(x, pattern=(1, 0, 2)) 
  return x

stacker = Lambda(stack_and_transpose)
outputs = stacker(outputs)

model = Model(
  inputs=[
    encoder_inputs_placeholder,
    decoder_inputs_placeholder,
    initial_s, 
    initial_c,
  ],
  outputs=outputs
)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

z = np.zeros((NUM_SAMPLES, LATENT_DIM_DECODER))

model.fit(
  [encoder_inputs, decoder_inputs, z, z], decoder_targets_one_hot,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_split=0.2
)

encoder_model = Model(encoder_inputs_placeholder, encoder_outputs)

encoder_outputs_as_input = Input(shape=(max_len_input, LATENT_DIM * 2,))
decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

context = one_step_attention(encoder_outputs_as_input, initial_s)

decoder_lstm_input = context_last_word_concat_layer([context, decoder_inputs_single_x])

o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[initial_s, initial_c])
decoder_outputs = decoder_dense(o)

decoder_model = Model(
  inputs=[
    decoder_inputs_single,
    encoder_outputs_as_input,
    initial_s, 
    initial_c
  ],
  outputs=[decoder_outputs, s, c]
)

idx2word_eng = {v:k for k, v in word2idx_inputs.items()}
idx2word_trans = {v:k for k, v in word2idx_outputs.items()}

def decode_sequence(input_seq):
  enc_out = encoder_model.predict(input_seq)

  target_seq = np.zeros((1, 1))
  
  target_seq[0, 0] = word2idx_outputs['<sos>']

  eos = word2idx_outputs['<eos>']

  s = np.zeros((1, LATENT_DIM_DECODER))
  c = np.zeros((1, LATENT_DIM_DECODER))

  output_sentence = []
  for _ in range(max_len_target):
    o, s, c = decoder_model.predict([target_seq, enc_out, s, c])
        
    idx = np.argmax(o.flatten())

    if eos == idx:
      break

    word = ''
    if idx > 0:
      word = idx2word_trans[idx]
      output_sentence.append(word)

    target_seq[0, 0] = idx

  return ' '.join(output_sentence)




while True:
  i = np.random.choice(len(input_texts))
  input_seq = encoder_inputs[i:i+1]
  translation = decode_sequence(input_seq)
  print('-')
  print('Input sentence:', input_texts[i])
  print('Predicted translation:', translation)
  print('Actual translation:', target_texts[i])

  ans = input("Continue? [Y/n]")
  if ans and ans.lower().startswith('n'):
    break