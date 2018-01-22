# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 14:59:07 2017

@author: Giancarlo
"""


from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed, Layer, Activation, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.layers.advanced_activations import ELU
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from scipy import spatial
import tensorflow as tf
import pandas as pd
import numpy as np
import codecs
import csv
import os



BASE_DIR = 'C:/Users/gianc/Desktop/PhD/Progetti/vae/'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'#'train_micro.csv'
GLOVE_EMBEDDING = BASE_DIR + 'glove.6B.50d.txt'
VALIDATION_SPLIT = 0.2
MAX_SEQUENCE_LENGTH = 15
MAX_NB_WORDS = 12000
EMBEDDING_DIM = 50



texts = [] 
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts.append(values[3])
        texts.append(values[4])
print('Found %s texts in train.csv' % len(texts))

#texts = texts[0:500]


#======================== Tokenize and pad texts lists ===================#
tokenizer = Tokenizer(MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index #the dict values start from 1 so this is fine with zeropadding
index2word = {v: k for k, v in word_index.items()}
#word_index = word_index[:tokenizer.num_words]
#word_index['unk'] = len(word_index)+1
print('Found %s unique tokens' % len(word_index))
sequences = tokenizer.texts_to_sequences(texts)
data_1 = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data_1.shape)
NB_WORDS = (min(tokenizer.num_words, len(word_index)) + 1 ) #+1 for zero padding and +1 for 'unk'


#==================== sample train/validation data =====================#
#np.random.seed(1234)
perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1)*(1-VALIDATION_SPLIT)):]

#data_1_train = np.vstack((data_1[idx_train]))
data_1_val = data_1[801000:807000] #np.vstack((data_1[idx_val]))
x_test_one_hot = np.zeros((data_1_val.shape[0], MAX_SEQUENCE_LENGTH, NB_WORDS))
x_test_one_hot[np.expand_dims(np.arange(data_1_val.shape[0]), axis=0).reshape(data_1_val.shape[0], 1), np.repeat(np.array([np.arange(MAX_SEQUENCE_LENGTH)]), data_1_val.shape[0], axis=0), data_1_val] = 1



def sent_generator(TRAIN_DATA_FILE, chunksize):
    reader = pd.read_csv(TRAIN_DATA_FILE, chunksize=chunksize, iterator=True)
    for df in reader:
        #print(df.shape)
        #df=pd.read_csv(TRAIN_DATA_FILE, iterator=False)
        val3 = df.iloc[:,3:4].values.tolist()
        val4 = df.iloc[:,4:5].values.tolist()
        flat3 = [item for sublist in val3 for item in sublist]
        flat4 = [str(item) for sublist in val4 for item in sublist]
        texts = [] 
        texts.extend(flat3[:])
        texts.extend(flat4[:])
        
        sequences = tokenizer.texts_to_sequences(texts)
        data_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        yield [data_train, data_train]





#======================== prepare GLOVE embeddings =============================#
#not yet used
embeddings_index = {}
f = open(GLOVE_EMBEDDING, encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

glove_embedding_matrix = np.zeros((NB_WORDS, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < NB_WORDS:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will the word embedding of unk.
            glove_embedding_matrix[i] = embedding_vector
        else:
            glove_embedding_matrix[i] = embeddings_index.get('unk')
print('Null word embeddings: %d' % np.sum(np.sum(glove_embedding_matrix, axis=1) == 0))

#NB_WORDS = (len(word_index) + 1



#====================== VAE model ============================================#

batch_size = 100
max_len = MAX_SEQUENCE_LENGTH
emb_dim = EMBEDDING_DIM
latent_dim = 32
intermediate_dim = 96
epsilon_std = 0.001
num_sampled=500
act = ELU()

#y = Input(batch_shape=(None, max_len, NB_WORDS))
x = Input(batch_shape=(None, max_len))
x_embed = Embedding(NB_WORDS, emb_dim, weights=[glove_embedding_matrix],
                            input_length=max_len, trainable=False)(x)
h = Bidirectional(LSTM(intermediate_dim, return_sequences=False, recurrent_dropout=0.2), merge_mode='concat')(x_embed)
#h = Bidirectional(LSTM(intermediate_dim, return_sequences=False), merge_mode='concat')(h)
h = Dropout(0.2)(h)
h = Dense(intermediate_dim, activation='linear')(h)
h = act(h)
h = Dropout(0.2)(h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
#z = Dropout(0.2)(z)
# we instantiate these layers separately so as to reuse them later
repeated_context = RepeatVector(max_len)
decoder_h = LSTM(intermediate_dim, return_sequences=True, recurrent_dropout=0.2)
decoder_mean = TimeDistributed(Dense(NB_WORDS, activation='linear'))#softmax is applied in the seq2seqloss by tf
h_decoded = decoder_h(repeated_context(z))
x_decoded_mean = decoder_mean(h_decoded)


# placeholder loss
def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)

#Sampled softmax
logits = tf.constant(np.random.randn(batch_size, max_len, NB_WORDS), tf.float32)
targets = tf.constant(np.random.randint(NB_WORDS, size=(batch_size, max_len)), tf.int32)
proj_w = tf.constant(np.random.randn(NB_WORDS, NB_WORDS), tf.float32)
proj_b = tf.constant(np.zeros(NB_WORDS), tf.float32)

def _sampled_loss(labels, logits):
    labels = tf.cast(labels, tf.int64)
    labels = tf.reshape(labels, [-1, 1])
    logits = tf.cast(logits, tf.float32)
    return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        proj_w,
                        proj_b,
                        labels,
                        logits,
                        num_sampled=num_sampled,
                        num_classes=NB_WORDS),
                    tf.float32)
softmax_loss_f = _sampled_loss


# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)
        self.target_weights = tf.constant(np.ones((batch_size, max_len)), tf.float32)

    def vae_loss(self, x, x_decoded_mean):
        #xent_loss = K.sum(metrics.categorical_crossentropy(x, x_decoded_mean), axis=-1)
        labels = tf.cast(x, tf.int32)
        xent_loss = K.sum(tf.contrib.seq2seq.sequence_loss(x_decoded_mean, labels, 
                                                     weights=self.target_weights,
                                                     average_across_timesteps=False,
                                                     average_across_batch=False), axis=-1)#,
                                                     #softmax_loss_function=softmax_loss_f), axis=-1)#,
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        print(x.shape, x_decoded_mean.shape)
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # we don't use this output, but it has to have the correct shape:
        return K.ones_like(x)

loss_layer = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, [loss_layer])
opt = Adam(lr=0.01) #SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
vae.compile(optimizer='adam', loss=[zero_loss])
vae.summary()


#======================= Model training ==============================#
def create_model_checkpoint(dir, model_name):
    filepath = dir + '/' + model_name + ".h5" #-{epoch:02d}-{decoded_mean:.2f}
    directory = os.path.dirname(filepath)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=False)
    return checkpointer

checkpointer = create_model_checkpoint('models', 'vae_seq2seq')


#train
#batch_size=200
# vae.fit(data_1_val, data_1_val,
        # shuffle=True,
        # epochs=5000,
        # batch_size=batch_size,
        # validation_data=(data_1_val, data_1_val), callbacks=[checkpointer])


#batch_size=512 #the actual batch_size is x2
nb_epoch=100
n_steps = 400000/batch_size#404000/batch_size
for counter in range(nb_epoch):
    print('-------epoch: ',counter,'--------')
    vae.fit_generator(sent_generator(TRAIN_DATA_FILE, batch_size/2),
                          steps_per_epoch=n_steps, epochs=1, callbacks=[checkpointer],
                          validation_data=(data_1_val, data_1_val))


vae.save('models/vae_lstm800k32dim96hid_ae.h5')
# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# build a generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(repeated_context(decoder_input))
_x_decoded_mean = decoder_mean(_h_decoded)
_x_decoded_mean = Activation('softmax')(_x_decoded_mean)
generator = Model(decoder_input, _x_decoded_mean)


index2word = {v: k for k, v in word_index.items()}
#batch_size=4
sent_encoded = encoder.predict(data_1_val, batch_size = 16)
x_test_reconstructed = generator.predict(sent_encoded)
#x_test_reconstructed = v.predict(x_test_encoded, batch_size=batch_size)


#test on a validation sentence
sent_idx = 672
reconstructed_indexes = np.apply_along_axis(np.argmax, 1, x_test_reconstructed[sent_idx])
np.apply_along_axis(np.max, 1, x_test_reconstructed[sent_idx])
np.max(np.apply_along_axis(np.max, 1, x_test_reconstructed[sent_idx]))
word_list = list(np.vectorize(index2word.get)(reconstructed_indexes))
word_list
original_sent = list(np.vectorize(index2word.get)(data_1_val[sent_idx]))
original_sent



#=================== Sentence processing and interpolation ======================#
# function to parse a sentence
def sent_parse(sentence, mat_shape):
    sequence = tokenizer.texts_to_sequences(sentence)
    padded_sent = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    return padded_sent#[padded_sent, sent_one_hot]


# input: encoded sentence vector
# output: encoded sentence vector in dataset with highest cosine similarity
def find_similar_encoding(sent_vect):
    all_cosine = []
    for sent in sent_encoded:
        result = 1 - spatial.distance.cosine(sent_vect, sent)
        all_cosine.append(result)
    data_array = np.array(all_cosine)
    maximum = data_array.argsort()[-3:][::-1][1]
    new_vec = sent_encoded[maximum]
    return new_vec


# input: two points, integer n
# output: n equidistant points on the line between the input points (inclusive)
def shortest_homology(point_one, point_two, num):
    dist_vec = point_two - point_one
    sample = np.linspace(0, 1, num, endpoint = True)
    hom_sample = []
    for s in sample:
        hom_sample.append(point_one + s * dist_vec)
    return hom_sample



# input: original dimension sentence vector
# output: sentence text
def print_latent_sentence(sent_vect):
    sent_vect = np.reshape(sent_vect,[1,latent_dim])
    sent_reconstructed = generator.predict(sent_vect)
    sent_reconstructed = np.reshape(sent_reconstructed,[max_len,NB_WORDS])
    reconstructed_indexes = np.apply_along_axis(np.argmax, 1, sent_reconstructed)
    np.apply_along_axis(np.max, 1, x_test_reconstructed[sent_idx])
    np.max(np.apply_along_axis(np.max, 1, x_test_reconstructed[sent_idx]))
    word_list = list(np.vectorize(index2word.get)(reconstructed_indexes))
    w_list = [w for w in word_list if w]
    print(' '.join(w_list))
    #print(word_list)
    
       
        
def new_sents_interp(sent1, sent2, n):
    tok_sent1 = sent_parse(sent1, [15])
    tok_sent2 = sent_parse(sent2, [15])
    enc_sent1 = encoder.predict(tok_sent1, batch_size = 16)
    enc_sent2 = encoder.predict(tok_sent2, batch_size = 16)
    test_hom = shortest_homology(enc_sent1, enc_sent2, n)
    for point in test_hom:
        print_latent_sentence(point)
        
print_latent_sentence(sent_encoded[35])
find_similar_encoding(sent_encoded[2])
print_latent_sentence(find_similar_encoding(sent_encoded[5]))

print_latent_sentence(sent_encoded[5])
print_latent_sentence(sent_encoded[6])
test_hom = shortest_homology(sent_encoded[2], sent_encoded[34], 10)
for point in test_hom:
    print_latent_sentence(point)

#====================== Example ====================================#
sentence1=['where can i find a book on machine learning']
mysent = sent_parse(sentence1, [15])
mysent_encoded = encoder.predict(mysent, batch_size = 16)
print_latent_sentence(mysent_encoded)
print_latent_sentence(find_similar_encoding(mysent_encoded))

sentence2=['how can i become a successful entrepreneur']
mysent2 = sent_parse(sentence2, [15])
mysent_encoded2 = encoder.predict(mysent2, batch_size = 16)
print_latent_sentence(mysent_encoded2)
print_latent_sentence(find_similar_encoding(mysent_encoded2))
print('-----------------')

new_sents_interp(sentence1, sentence2, 6)
