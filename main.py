from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
import numpy as np
from  EERNNDataProcessor import EERNNDataProcessor
import tensorflow as tf
import os


# 模型定义部分
class EERNN(tf.keras.Model):
    def __init__(self,  embedding_matrix, onehot_matrix):
        super(EERNN, self).__init__()
        self.lstm = tf.keras.layers.LSTM(name="lstm", units=LSTM_UNITS, return_sequences=True, return_state=False,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.1549193338, maxval=0.1549193338))

        self.dropout = tf.keras.layers.Dropout(0.1)

        self.dense1 = tf.keras.layers.Dense(name='dense1', units=50, activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.1732050808, maxval=0.1732050808))

        self.dense2 = tf.keras.layers.Dense(name='dense2', units=1,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.3429971703,maxval=0.3429971703))

        self.sotfmax = tf.keras.layers.Softmax()

        self.embedding = tf.keras.layers.Embedding(input_dim=5111, output_dim=EMBEDDING_DIM, name="embedding", weights=[embedding_matrix], trainable=False)

        self.embedding2 = tf.keras.layers.Embedding(input_dim=2, output_dim=4*LSTM_UNITS, name="embedding2", weights=[onehot_matrix], trainable=False)

        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(name="bi_lstm", units=LSTM_UNITS, return_sequences=True, return_state=False,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.2, maxval=0.2),
            kernel_regularizer=tf.keras.regularizers.l2(0.00004)))



    def call_encode(self, pid2seq):
        # 将题目文本进行嵌入
        # pid2seq :[483, 100]
        # x: [483, 100, 50]
        x = self.embedding(pid2seq)
        if pid2seq.shape[0]<2000:
            x = self.bi_lstm(x)
        else:
            x1 = self.bi_lstm(x[0:2000, :, :])
            x2 = self.bi_lstm(x[2000:4000, :, :])
            x3 = self.bi_lstm(x[4000: ,:, :])
            x = tf.concat([x1, x2, x3], axis=0)
        x = tf.math.reduce_max(x, axis=1)
        sim = self.cosine_distance(x)
        return x, sim 

    def cosine_distance(self, inputs):
        num_vec = inputs.shape[0]
        x = tf.expand_dims(inputs, axis=1)
        x = tf.tile(x, [1, num_vec, 1])
        y = tf.expand_dims(inputs, axis=0)
        y = tf.tile(y, [num_vec, 1, 1])
        sim = -1 * tf.keras.losses.cosine_similarity(x, y)
        return sim

    def cal_hatt(self, ht, pro_id_ont_hot, X, cos_X, trimatrix):
        # 每一行为一个题目与其他题目向量的余弦值
        a = tf.matmul(pro_id_ont_hot, cos_X)
        aj = tf.expand_dims(a, -1)
        hj = tf.expand_dims(ht, 2)
        hidden = tf.multiply(aj, hj)
        ajhj = tf.tensordot(hidden, trimatrix, [[1],[0]])
        ajhj = tf.transpose(ajhj, [0, 3, 1, 2]) 
        x1 = tf.expand_dims(X, 0)
        x1 = tf.expand_dims(x1, 0)
        XX = tf.tile(x1, multiples=[BATCH_SIZE, pro_id_ont_hot.shape[1], 1, 1])
        hatt = tf.concat([ajhj , XX], -1)
        return hatt

    def call(self, data, num_pro, X, cos_X, trimatrix):
        pro_id, label = data
        pro_id_one_hot = tf.one_hot(pro_id, num_pro)
        label_embedding = self.embedding2(label)
        # [batch, 题目序列， 题目词向量表达]
        t_X = tf.matmul(pro_id_one_hot, X)
        t_X = tf.tile(t_X, [1, 1, 2])
        # [batch, 题目序列， 题目词向量与0,1拼接]
        xt = tf.multiply(t_X, label_embedding)
        ht = self.lstm(xt)
        hatt= self.cal_hatt(ht, pro_id_one_hot, X, cos_X, trimatrix)
        r = self.dense1(hatt)
        prediction = self.dense2(r)
        return prediction


# 损失函数
def entroy_loss(prediction, data):
    target_id, target_correctness = data

    # 真实值
    target_correctness = target_correctness[:, 1:]
    flat_target_correctness = tf.reshape(target_correctness, [-1])
    flat_target_correctness = tf.cast(flat_target_correctness, dtype=tf.float32)

    # 预测值
    target_id = target_id[:, 1:]
    num_pro = prediction.shape[-2]

    # 计算偏移量
    flat_bias_target_id = num_pro * tf.range(BATCH_SIZE * target_id.shape[-1])
    # 获取预测数据所在ID
    flat_target_id = tf.reshape(target_id, [-1]) + flat_bias_target_id
    prediction = prediction[:, :-1, :, :]
    flat_logits = tf.reshape(prediction, [-1])
    flat_target_logits = tf.gather(flat_logits, flat_target_id)

    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_target_correctness, logits=flat_target_logits))
    return loss

def train(DataName,TmpDir):
    trimatrix = np.tri(MAXLEN, MAXLEN, 0).T
    trimatrix = tf.cast(trimatrix, tf.float32)
    # 定义数据处理器
    data_processor = EERNNDataProcessor([15,1000000,0.06,1],[10,1000000,0.02,1],['2005-01-01 23:47:31','2019-01-02 11:21:49'],True,DataName,TmpDir)
    # 获取处理好的数据
    pid2seq, embedding_matrix, dataset, onehot_matrix = data_processor.LoadEERNNData(BATCH_SIZE, PREFETCH_SIZE, SHUFFLE_BUFFER_SIZE, LSTM_UNITS,100)
    # 定义模型
    model = EERNN(embedding_matrix, onehot_matrix)
    # 周期数
    epochs = 10
    # 学习率
    lr = 0.01
    # 学习率衰减率
    lr_decay = 0.92
    print("Start training...")
    for epoch in range(epochs):
        optimizer = tf.keras.optimizers.Adam(lr * lr_decay ** epoch)
        # 打乱数据集
        dataset.shuffle(BUFFER_SIZE)
        for batch, data in enumerate(dataset):
            loss = 0
            with tf.GradientTape() as tape:
                # pid2seq　保存了每道题目对应的词语ID　[pro_num, word_num]
                X,cos_X =  model.call_encode(pid2seq)
                # X 记录了每道题目对应的hatten
                # cos_X 记录了每道题目对应的hatten之间的cosine
                # 计算预测值
                prediction = model(data, pid2seq.shape[0], X, cos_X, trimatrix)
                # 计算损失值
                loss += entroy_loss(prediction, data)
            grad = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
            # 打印该批次损失
            batch_loss = (loss / BATCH_SIZE)
            if batch%100 == 0:
                print("Epoch {} Batch {} Loss {:.4f}".format(epoch + 1, batch, batch_loss.numpy()))

if __name__=='__main__':
    NUM_WORDS = 5000
    BUFFER_SIZE = 12
    BATCH_SIZE = 8
    EMBEDDING_DIM = 50
    LSTM_UNITS = 50
    PREFETCH_SIZE = 128
    SHUFFLE_BUFFER_SIZE = 5120
    MAXLEN = 100
    data_name = 'hdu'
    tmp_dir = './data/'
    train(data_name, tmp_dir)



