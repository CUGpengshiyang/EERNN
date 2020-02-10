from OJDataProcessor import OJDataProcessor
from OJProblemtextProcessor import OJProblemtextProcessor
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import os

class EERNNDataProcessor(object):
    def __init__(self, userLC, problemLC, timeLC, OnlyRight, DataName, TmpDir):
        self.OJData = OJDataProcessor(DataName, TmpDir)
        self.OJData.LoadSubmitRecordData(userLC, problemLC, timeLC, OnlyRight)
        self.OJProblem = OJProblemtextProcessor(userLC, problemLC, timeLC, OnlyRight, DataName, TmpDir)

    def LoadEERNNData(self, BATCH_SIZE, PREFETCH_SIZE, SHUFFLE_BUFFER_SIZE, LSTM_UNITS, maxLen=100):
        # submitRecord 每个序列为一个用户的提交序列
        np_sample = len(self.OJData.submitRecord)
        seqs_ans = {}
        for i in range(len(self.OJData.submitRecord)):
            if (len(self.OJData.submitRecord[i]) <= maxLen):
                seqs_ans[i] = self.OJData.submitRecord[i]
            else:
                # 获取原有序列的长度
                # 超过maxlen的部分需要进行截断
                # 并添加到后面
                #  
                nowId = len(self.OJData.submitRecord[i])
                seqs_ans[i] = self.OJData.submitRecord[i][nowId - maxLen:nowId]
                nowId -= maxLen
                while (nowId >= 0):
                    if (nowId - maxLen >= 0):
                        seqs_ans[np_sample] =self.OJData.submitRecord[i][nowId - maxLen:nowId]
                    else:
                        seqs_ans[np_sample] = self.OJData.submitRecord[i][0:maxLen]
                    np_sample += 1
                    nowId -= maxLen
        pro_dic,cor_dic = [],[]
        # pro_dic, 用于保存题目ID
        # cor_dic, 用于保存题目对应的回答结果
        for _, it in seqs_ans.items():
            pro, cor = zip(*it)
            pro_dic.append(list(pro))
            cor_dic.append(list(cor))
        # 未到最大值，在最后序列填充0
        tmp_pro = tf.keras.preprocessing.sequence.pad_sequences(pro_dic, value=0, padding='post', truncating='post',maxlen=maxLen)
        tmp_cor = tf.keras.preprocessing.sequence.pad_sequences(cor_dic, value=0, padding='post', truncating='post', maxlen=maxLen)
        train_data = np.concatenate((tmp_pro, tmp_cor), axis=1)
        # 题目ID
        dataset_train_pro = tf.data.Dataset.from_tensor_slices(train_data[:, :maxLen])
        # 对应提交0,1
        dataset_train_cor = tf.data.Dataset.from_tensor_slices(train_data[:, maxLen:])
        # 每个元素为 [[题目ID], [对应提交]]
        dataset_train = tf.data.Dataset.zip((dataset_train_pro, dataset_train_cor))
        dataset_train = dataset_train.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        onehot_matrix = np.zeros((2, 4 * LSTM_UNITS))
        onehot_matrix[0] = [0] * 2 * LSTM_UNITS + [1] * 2 * LSTM_UNITS
        onehot_matrix[1] = [1] * 2 * LSTM_UNITS + [0] * 2 * LSTM_UNITS
        # 题目文本处理
        pro_dic, embedding_matrix = self.OJProblem.Problem2Tensor()
        return  pro_dic, embedding_matrix, dataset_train,  onehot_matrix



if __name__ == "__main__":
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
    DataProssor = EERNNDataProcessor([15,1000000,0.06,1], [10,1000000,0.02,1], ['2005-01-01 23:47:31','2019-01-02 11:21:49'], True, data_name, tmp_dir)
    # 获取处理好的数据
    pro_dic, embedding_matrix, dataset, embedding_matrix2 = DataProssor.LoadEERNNData(BATCH_SIZE, PREFETCH_SIZE, SHUFFLE_BUFFER_SIZE, LSTM_UNITS, 100)
    print("END")
