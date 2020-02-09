from hyper_params import *
from Decoder import *
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
import time
import numpy as np
from  EERNNDataProcessor import EERNNDataProcessor
import tensorflow as tf
import os

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
    decoder = Decoder(embedding_matrix, embedding_matrix2)
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
                X,cos_X =  decoder.call_encode(pro_dic)
                # 计算预测值
                prediction = decoder(data,pro_dic.shape[0],X,cos_X,trimatrix)
                # 计算损失值
                loss += entroy_loss(prediction, data)
            gradients = tape.gradient(loss, decoder.trainable_variables)
            optimizer.apply_gradients(zip(gradients, decoder.trainable_variables))
            # 打印该批次损失
            batch_loss = (loss / int(data_target.shape[1]))
            if batch%100 == 0:
                print("Epoch {} Batch {} Loss {:.4f}".format(epoch + 1, batch, batch_loss.numpy()))
            os._exit(0)
        end = time.time()
        # 保存模型参数
        decoder.save_weights('./model/my_model_'+str(epoch+1))
        # 打印单个周期的时间消耗
        print("Epoch {} cost {}".format(epoch + 1, end - start))

if __name__=='__main__':
    train('hdu',"./Data/datapart/")
