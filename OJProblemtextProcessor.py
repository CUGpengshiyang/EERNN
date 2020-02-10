import tensorflow as tf
import os
import re
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords as pw

class OJProblemtextProcessor(object): #数据集名称
    DataName = ""
    #临时文件夹位置
    TmpDir = ""
    word2Id = {}
    problemName2problemId = {}
    problemId2problemName = {}
    problemList = []
    def __init__(self, userLC, problemLC, timeLC, OnlyRight, DataName='hdu',TmpDir='./data/'):
        self.DataName = DataName
        self.TmpDir = TmpDir
        self.LC2Str = ('userLC_' + str(userLC) + '_problemLC_' + str(problemLC) + '_timeLC_' + re.sub(r':', '.', str(timeLC)) + '_OnlyRight_' + str(OnlyRight))
        self.input_problem_path = self.TmpDir + self.DataName + '_problem.txt'
        self.filter_problem_path = self.TmpDir + self.DataName +self.LC2Str+ '_filtered_'+'problem.txt'
        self.embedding_file = self.TmpDir + 'glove.6B.'
        self.word2IdPath = self.TmpDir + self.DataName+'_word2Id_' +self.LC2Str+'.txt'
        self.cantembedding = self.TmpDir +'cantembedding.txt'
        # 加载数据
        with open(self.TmpDir + self.DataName + self.LC2Str+ '_problemName2problemId.pkl', 'rb') as fr:
            self.problemName2problemId = pickle.load(fr)
        with open(self.TmpDir + self.DataName + self.LC2Str + '_problemId2problemName.pkl', 'rb') as fr:
            self.problemId2problemName = pickle.load(fr)


    def Sentence2Words(self, raw_str):
        # re_sentence
        raw_str = re.sub(r'([+])',' add ',raw_str)
        raw_str = re.sub(r'([-])',' sub ',raw_str)
        raw_str = re.sub(r"([?.!,])", r" \1 ", raw_str)
        # #数学公式在处理时就没有了 而且也没法预训练  公式处理程序
        raw_str = re.sub(r'[" "]+', " ", raw_str)
        raw_str = re.sub(r"[^a-zA-Z0-9]+", " ", raw_str)
        raw_str = raw_str.strip()
        # split2word
        sentence = ' '.join([word.lower() for word in str(raw_str).split() if word.lower() not in self.stopword])
        return sentence


    def SortandInitialProblembyId(self):
        """初始化 problemList"""
        if not os.path.exists(self.filter_problem_path):
            print(self.filter_problem_path + ' has exist')
            with open(self.filter_problem_path, 'r') as f:
                for line in f:
                    self.problemList.append(line.strip())
        else:
            # 构造stopword列表
            self.stopword = pw.words('english')
            with open(self.cantembedding, 'r') as f:
                for word in f:
                    self.stopword.append(word.strip())

            pid2seq = dict()
            index = 1000
            # problemName 题目的真实ID
            # problemId   映射的题目ID
            with open(self.input_problem_path, 'r', encoding='gb18030') as f:
                for line in f:
                    if index in self.problemName2problemId.keys():
                        pid2seq[self.problemName2problemId[index]]=self.Sentence2Words(line)
                    index +=1
            pid2seq = sorted(pid2seq.items(),key=lambda x:x[0])
            w_f = open(self.filter_problem_path, 'w')
            w_f.writelines('\n')
            self.problemList.append([])
            for k, v in pid2seq:
                if len(v)==0:
                    print('has empty problem', self.problemId2problemName[k])
                else:
                    self.problemList.append(v)
                    w_f.writelines(v+'\n')
            w_f.close()

    def LoadGlove(self,EMBEDDING_DIM):
        """导入词向量"""
        embedding_file = self.embedding_file + str(EMBEDDING_DIM) + 'd.txt'
        word2vec = dict() # 定义字典
        with  open(embedding_file, 'r', encoding='utf8') as fr:
            for line in fr:
                values = line.split()
                word = values[0]
                word2vec[word] = np.array(values[1:], dtype='float32')


        # 转化为矩阵：构建可以加载到embedding层中的嵌入矩阵，形为(max_words（单词数）, embedding_dim（向量维数）)
        embedding_matrix = np.zeros((len(self.word2Id), EMBEDDING_DIM))
        for word, idx in self.word2Id.items():  # 字典里面的单词和索引
            # 这里只考虑了之前预训练过的词 那些没有的忽略了
            if word in word2vec:
                embedding_matrix[idx] = word2vec[word]
        return embedding_matrix

    def Problem2Tensor(self,EMBEDDING_DIM=50,Maxlen=100):
        # produces pro_dic,embedding_matrix without labels used for EERNN
        self.SortandInitialProblembyId()
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=4000,split=" ",char_level=False)
        tokenizer.fit_on_texts(self.problemList)
        tokenizer.word_index['<pad>'] = 0
        # 由整数索引的向量
        pid2seq = tokenizer.texts_to_sequences(self.problemList)  
        # 将每个序列都格式化为100
        # 第一维度 题目数量; 第二维度 题目文本长度
        pid2seq = tf.keras.preprocessing.sequence.pad_sequences(pid2seq, padding='post', truncating='post',maxlen=Maxlen)
        
        # 保存索引字典
        self.word2Id = tokenizer.word_index 
        with open(self.word2IdPath,'w') as f:
            for k, v in  self.word2Id.items():
                f.writelines(str(k)+' '+str(v)+'\n')

        # 加载glove向量
        embedding_matrix =  self.LoadGlove(EMBEDDING_DIM)

        return pid2seq, embedding_matrix


if __name__ == "__main__":
    k = OJProblemtextProcessor([15,1000000,0.06,1],[10,1000000,0.02,1],['2005-01-01 23:47:31','2019-01-02 11:21:49'],OnlyRight=True)
    pid2seq, embed_matrix = k.Problem2Tensor()
    print(pid2seq.shape)
    print(embed_matrix.shape)