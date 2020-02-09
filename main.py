from hyper_params import *
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
import time
import numpy as np
from  EERNNDataProcessor import EERNNDataProcessor
import tensorflow as tf
import os


# 模型定义部分
class EERNN(tf.keras.Model):
    def __init__(self,embedding_matrix,embedding_matrix2):
        super(EERNN, self).__init__()
        # LSTM 
        self.lstm = tf.keras.layers.LSTM(name="lstm", units=LSTM_UNITS, return_sequences=True, return_state=False,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.1549193338, maxval=0.1549193338))

        self.dropout = tf.keras.layers.Dropout(0.1)

        self.dense1 = tf.keras.layers.Dense(name='dense1', units=50, activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.1732050808, maxval=0.1732050808))

        self.dense2 = tf.keras.layers.Dense(name='dense2', units=1,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.3429971703,maxval=0.3429971703))

        self.sotfmax = tf.keras.layers.Softmax()

        self.embedding = tf.keras.layers.Embedding(input_dim=5111, output_dim=EMBEDDING_DIM, name="embedding", weights=[embedding_matrix], trainable=False)

        self.embedding2 = tf.keras.layers.Embedding(input_dim=2, output_dim=4*LSTM_UNITS, name="embedding2", weights=[embedding_matrix2], trainable=False)

        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(name="bi_lstm", units=LSTM_UNITS, return_sequences=True, return_state=False,
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


# 模型训练部分
def cal_flat_target_logits(prediction,target_id,target_correctness):
    num_pro = prediction.shape[-2]
    prediction,target_id, target_correctness = prediction[:,:-1,:,:],target_id[:,1:],target_correctness[:,1:]
    flat_logits = tf.reshape(prediction, [-1])
    flat_target_correctness = tf.reshape(target_correctness, [-1])
    flat_bias_target_id = num_pro * tf.range(BATCH_SIZE * target_id.shape[-1])
    flat_target_id = tf.reshape(target_id, [-1])+flat_bias_target_id
    flat_target_logits = tf.gather(flat_logits, flat_target_id)
    return flat_target_logits,flat_target_correctness

# 损失函数
def entroy_loss(prediction, data):
    target_id, target_correctness = data
    flat_target_logits, flat_target_correctness = cal_flat_target_logits(prediction, target_id, target_correctness)
    flat_target_correctness = tf.cast(flat_target_correctness,dtype=tf.float32)
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_target_correctness, logits=flat_target_logits))

def train(DataName,TmpDir):
    trimatrix = np.tri(MAXLEN, MAXLEN, 0).T
    trimatrix = tf.cast(trimatrix, tf.float32)
    # 定义数据处理器
    DataProssor = EERNNDataProcessor([15,1000000,0.06,1],[10,1000000,0.02,1],['2005-01-01 23:47:31','2019-01-02 11:21:49'],True,DataName,TmpDir)
    # 获取处理好的数据
    pro_dic, embedding_matrix, dataset, _, embedding_matrix2 = DataProssor.LoadEERNNData(BATCH_SIZE, PREFETCH_SIZE, SHUFFLE_BUFFER_SIZE, LSTM_UNITS,100)
    # 周期数
    epochs = 10
    # 定义模型
    model = EERNN(embedding_matrix, embedding_matrix2)
    # 学习率
    lr = 0.01
    # 学习率衰减率
    lr_decay = 0.92
    print("Start training...")
    for epoch in range(epochs):
        optimizer = tf.keras.optimizers.Adam(lr * lr_decay ** epoch)
        start = time.time()
        # 打乱数据集
        dataset.shuffle(BUFFER_SIZE)
        for batch, data in enumerate(dataset):
            data_target, _ = data
            loss = 0
            with tf.GradientTape() as tape:
                X,cos_X =  model.call_encode(pro_dic)
                # 计算预测值
                prediction = model(data,pro_dic.shape[0],X,cos_X,trimatrix)
                # 计算损失值
                loss += entroy_loss(prediction, data)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            # 打印该批次损失
            batch_loss = (loss / int(data_target.shape[1]))
            if batch%100 == 0:
                print("Epoch {} Batch {} Loss {:.4f}".format(epoch + 1, batch, batch_loss.numpy()))
            os._exit(0)
        end = time.time()
        # 保存模型参数
        model.save_weights('./model/my_model_'+str(epoch+1))
        # 打印单个周期的时间消耗
        print("Epoch {} cost {}".format(epoch + 1, end - start))

if __name__=='__main__':
    train('hdu',"./data/")
