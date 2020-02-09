from hyper_params import *

class Decoder(tf.keras.Model):
    def __init__(self,embedding_matrix,embedding_matrix2):
        super(Decoder, self).__init__()
        self.lstm = tf.compat.v1.keras.layers.CuDNNLSTM(name="lstm", units=LSTM_UNITS, return_sequences=True, return_state=False,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.1549193338, maxval=0.1549193338))
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.dense1 = tf.keras.layers.Dense(name='dense1', units=50, activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.1732050808, maxval=0.1732050808))
        self.dense2 = tf.keras.layers.Dense(name='dense2', units=1,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.3429971703,maxval=0.3429971703))
        self.sotfmax = tf.keras.layers.Softmax()
        self.embedding = tf.keras.layers.Embedding(
            #input_dim=NUM_WORDS,
            input_dim=5111,
            output_dim=EMBEDDING_DIM,
            name="embedding",
            weights=[embedding_matrix],
            trainable=False,
        )
        self.embedding2 = tf.keras.layers.Embedding(
            input_dim=2,
            output_dim=4*LSTM_UNITS,
            name="embedding2",
            weights=[embedding_matrix2],
            trainable=False,
        )
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(
            name="bi_lstm",
            units=LSTM_UNITS,
            return_sequences=True,
            return_state=False,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.2, maxval=0.2),
            kernel_regularizer=tf.keras.regularizers.l2(0.00004),
        ))	

    def call_encode(self, pro_dic):
        x = self.embedding(pro_dic)
        if pro_dic.shape[0]<2000:
            x = self.bi_lstm(x)
        else:
            x1 = self.bi_lstm(x[0:2000, :, :])
            x2 = self.bi_lstm(x[2000:4000, :, :])
            x3 = self.bi_lstm(x[4000: ,:, :])
            x = tf.concat([x1, x2, x3], axis=0)
        x = tf.compat.v1.reduce_max(x, axis=1, keep_dims=False, name=None)
        cos_X = self.cosine(x, x)
        return x, cos_X

    def cosine(self, q, a):
        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(q), 1))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(a), 1))
        pooled_len_1 = tf.expand_dims(pooled_len_1, axis=0)
        pooled_len_2 = tf.expand_dims(pooled_len_2, axis=-1)
        norm_matrix = tf.tensordot(pooled_len_2, pooled_len_1, [[1], [0]])
        dot_matrix = tf.tensordot(a, q, [[1], [1]])
        sim = dot_matrix / norm_matrix
        return sim

    def call(self, data,num_pro,X,cos_X,trimatrix):
        data_target, data_cor = data
        data_target_one_hot = tf.one_hot(data_target, num_pro)
        data_cor_embedding = self.embedding2(data_cor)
        t_X = tf.tensordot(data_target_one_hot, X, [[2], [0]])
        t_X = tf.concat([t_X, t_X], axis=2)
        xt = tf.multiply(t_X, data_cor_embedding)

        ht = self.lstm(xt)
        hatt= self.cal_hatt(ht,data_target_one_hot,X,cos_X,trimatrix)
        r = self.dense1(hatt)
        r = self.dense2(r)
        return r

    def cal_hatt(self,ht,data_target_one_hot,X,cos_X,trimatrix):
        a = tf.tensordot(data_target_one_hot,cos_X,[[2],[0]])
        aj = tf.expand_dims(a,-1)
        hj = tf.expand_dims(ht,2)
        hidden = tf.multiply(aj,hj)
        ajhj = tf.tensordot(hidden,trimatrix,[[1],[0]])
        ajhj = tf.transpose(ajhj, [0, 3, 1,2])
        x1 = tf.expand_dims(X, 0)
        x1 = tf.expand_dims(x1, 0)
        XX = tf.tile(x1, multiples=[BATCH_SIZE,data_target_one_hot.shape[1], 1, 1])
        hatt = tf.concat([ajhj , XX],-1)
        return hatt


