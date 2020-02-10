import os
import numpy as np
import pandas as pd
import pickle
import re
# coding = utf-8
'''
需要给定原始的OJ数据集
原始数据集包含一下两部分
    1.用户的签到记录（xxx_RawSubmitRecord.txt）
        每行为一条提交记录,具体包含一下属性
            提交ID,提交时间,返回状态,题目编号,执行时间,内存占用,代码长度,用户名
            以上每一行不同属性以3个空格分割
    2.知识点题目映射关系（xxx_RawKnowledge2Problem.txt）
        每行为某一知识点的包含的题目，具体形式如下(n表示某个知识点包含题目的数量)
            知识点名:题目ID1,题目ID2,......,题目IDn
        以上每一行具体表示为以下形式:
            knowledgeName:ProblemID1,ProblemID2,ProblemID3,...,ProblemIDn
'''
class OJDataProcessor(object):
    #用户与用户ID之间的转换关系，字典中每个用户都是合法的
    userName2userId = {}
    userId2userName = {}
    #题目和题目ID之间的转换关系，字典中每一道题目都是合法的
    problemName2problemId = {}
    problemId2problemName = {}
    #合法的所有提交记录
    submitRecord = []
    #题目-知识点矩阵
    QMatrix = []
    #时间-用户-题目tensor
    SubmitTensor = []
    #数据集名称
    DataName = ""
    #临时文件夹位置
    TmpDir = ""

    def __init__(self,DataName = 'hdu',TmpDir = './data/'):
        self.DataName = DataName
        self.TmpDir = TmpDir
        self.Knowledge2ProblemPath = self.TmpDir + self.DataName + '_RawKnowledge2Problem.txt'
        self.RawSubmitRecordPath = self.TmpDir + self.DataName + '_RawSubmitRecord.txt'
        print('Knowledge2ProblemPath:',self.Knowledge2ProblemPath,'\nRawSubmitRecordPath:', self.RawSubmitRecordPath)
        self.hdu_partproblem =self.TmpDir + self.DataName+'_partproblem.txt'
        self.proid = []
        with open(self.hdu_partproblem,'r') as f:
            for line in f:
                self.proid = line.strip().split(',')
        self.RawSubmitRecord2CSV()
        self.RawKnowledge2CSV()

    #将原始提交记录文件转化为CSV文件方便后期处理 
    #转化后名称为"xxx_RawSubmitRecord.csv"
    #过程中要加入判断当前文件是否存在，存在不用进行二次处理
    #输出提示是否需要重新计算
    #如果处理的话,加上处理的进度条yy
    def RawSubmitRecord2CSV(self):
        if os.path.exists(self.TmpDir + self.DataName+'_RawSubmitRecord.csv'):
            return
        print('is RawSubmitRecord2CSV')
        count=0

        thefile=open(self.RawSubmitRecordPath, 'r', encoding='gb18030', errors='ignore')
        while True:
            buffer=thefile.read(1024*8192)
            if not buffer:
                break
            count+=buffer.count('\n')
        thefile.close()

        w_f = open(self.TmpDir + self.DataName+'_RawSubmitRecord.csv', 'w', encoding='utf-8')
        num_line = 0
        with open(self.RawSubmitRecordPath, 'r', encoding='gb18030', errors='ignore') as f:
            for line in f:
                num_line += 1
                #if num_line %1000000 == 0:
                    #print('has loading line :',num_line,'/',count);
                v = line.split('   ')
                if len(v) != 10:
                    continue
                v[8]= re.sub(r',+','_',v[8])
                out = v[8].strip()
                for i in range(1, 8):
                    out += ',' + v[i].strip()
                out += '\r'
                w_f.writelines(out)
        print('LoadSubmitRecordData end')
    #将原始知识点题目映射文件转化为CSV文件方便后期处理
    #转化后名称为"xxx_Knowledge2Problem.csv"3
    #过程中要加入判断当前文件是否存在，存在不用进行二次处理
    #输出提示是否需要重新计算
    #如果处理的话,加上处理的进度条
    def RawKnowledge2CSV(self):
        pass

    #userLC = [最少提交次数，最多提交次数，最低通过率，最高通过率]
    #ProblemLC = [最少提交次数，最多提交次数，最低通过率，最高通过率]
    #timeLC = [起始时间（单位秒），终止时间（秒）]
    #LC2Str将限制条件映射为一个字符串
    def LC2Str(self,userLC,problemLC,timeLC,OnlyRight):
        c = re.sub(r':', '.', str(timeLC))
        return ('userLC_'+str(userLC)+'_problemLC_'+str(problemLC)+'_timeLC_'+c+'_OnlyRight_'+str(OnlyRight))

    #userLC = [最少提交次数，最多提交次数，最低通过率，最高通过率]
    #ProblemLC = [最少提交次数，最多提交次数，最低通过率，最高通过率]
    #timeLC = [起始时间（单位秒），终止时间（秒）]
    # 当OnlyRight为真的时候，只考虑Accepted，其它所有情况划分为一类，等OnlyRight为假的时候
    # 分为这几种情况dic_status = {'Accepted': 1, 'Wrong Answer': 0, 'Time Limit Exceeded': 2, 'other': 3}
    #根据限制条件过滤数据
    #最终要获得userName-userId,userId-userName,problemRawId-problemId,problemId-ProblemRawId四个字典，要保证存在在字典中的每一个题目与用户都是满足限制条件的，以及满足限制条件的提交记录。
    #加进度条并保持求的5个结果,最终结果保持在TmpDir路径下
    #五个文件的前缀应该是DataName+LC2Str(userLC,problemLC,timeLC),五个文件各自的后缀名称你可以自行设计
    def FilterSubmitRecordData(self,userLC,problemLC,timeLC,OnlyRight):
        print('is FilterSubmitRecordData')
        dic_status = {'Accepted': 1, 'Wrong Answer': 0, 'Time Limit Exceeded': 2, 'other': 3}
        def convert_statue(value):
            value = value.strip()
            if OnlyRight:
                if value == 'Accepted':
                    return 1
                else:
                    return 0
            else:
                if value in dic_status.keys():
                    return dic_status[value]
                else:
                    return dic_status['other']

        def delet_pro(df,filter_id,filter_condition):
            def cal_pro(v):
                total_num = v[filter_id].count()
                if filter_id == 'name':
                    ac_num = v[v['status'] == 1]['PID'].nunique()
                else:
                    ac_num = v[v['status'] == 1]['PID'].count()
                v['total'] = total_num
                v['ac_num'] = ac_num
                v['pro'] = ac_num / total_num
                return v[[filter_id, 'total', 'ac_num', 'pro']]

            judge = df.groupby(filter_id).apply(cal_pro)
            judge.drop_duplicates(subset=[filter_id], inplace=True)
            judge = judge[(judge['pro'] <filter_condition[2]) | (judge['pro'] >filter_condition[3])|(judge['total'] < filter_condition[0]) |(judge['total'] >filter_condition[1])]
            return judge


        df = pd.read_csv(self.TmpDir + self.DataName+'_RawSubmitRecord.csv', delimiter=',', lineterminator='\r', header=None, index_col=False)
        df.columns = ['name', 'sub_time', 'status', 'PID', 'op_time', 'memeory', 'codeSize', 'lan']
        #filter time
        df = df[(df['sub_time']>=timeLC[0]) & (df['sub_time']<=timeLC[1])]
        #status
        df['status'] = df['status'].apply(convert_statue)

        df = df[df['PID'].isin(self.proid)]
        #filter problem return need delete problem
        print('is filtering problem\nthe filtered problem list is ')
        judge_pro = delet_pro(df,'PID',problemLC)
        df = df[-df['PID'].isin(list(judge_pro['PID']))]
        empty = [1184, 1499, 1670, 1737, 1739, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1762, 1763, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1784, 1830, 1831, 1832, 1834, 1835, 1910, 2038, 2958, 4905, 6265, 6266, 6267, 6268, 6269, 6270, 6271, 6272, 6273, 6274, 6275]
        df = df[-df['PID'].isin(empty)]
        #filter student return need delete student
        print('is filtering student \nthe filtered student list is')
        judge_stu = delet_pro(df,'name',userLC)
        df = df[-df['name'].isin(list(judge_stu['name']))]
        #filter end
        print('is writing to file')
        t = self.LC2Str(userLC,problemLC,timeLC,OnlyRight)
        w_f1 = open(self.TmpDir + self.DataName+
               t+ '_userName2userId.pkl', 'wb' )
        w_f2 = open(self.TmpDir + self.DataName+
                t+ '_userId2userName.pkl', 'wb')
        w_f3 = open(self.TmpDir + self.DataName+
               t+ '_problemName2problemId.pkl', 'wb' )
        w_f4 = open(self.TmpDir + self.DataName+
                t+ '_problemId2problemName.pkl', 'wb')

        def tonameid(name):
            if name not in self.userName2userId.keys():
                id = len(self.userName2userId)
                self.userName2userId[name] = id
                self.userId2userName[id] = name
            return self.userName2userId[name]
        def toproid(name):
            if name not in self.problemName2problemId.keys():
                id = len(self.problemName2problemId)
                self.problemName2problemId[name] = id
                self.problemId2problemName[id] = name
            return self.problemName2problemId[name]
        df['name'] =df['name'].apply(tonameid)
        df['PID'] = df['PID'].apply(toproid)
        df = df.sort_values(by=['name', 'sub_time', 'PID'])
        print(df)
        df[['name','PID','status']].to_csv(self.TmpDir + self.DataName+self.LC2Str(userLC,problemLC,timeLC,OnlyRight) + '_EERNN_input.csv', index=False, header=False)
        pickle.dump(self.userName2userId, w_f1, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.userId2userName, w_f2, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.problemName2problemId, w_f3, pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.problemId2problemName, w_f4, pickle.HIGHEST_PROTOCOL)
        w_f1.close(),w_f2.close(),w_f3.close(),w_f4.close()
        print('FilterSubmitRecordData end')

    #导入满足限制条件的数据，
    #如果之前没有处理过直接调用FilterSubmitRecordData函数,如果处理过直接读取之前结果，
    #构建4个dict以及合法的提交记录list。
    def LoadSubmitRecordData(self,userLC,problemLC,timeLC,OnlyRight):
        prefix = self.TmpDir + self.DataName+self.LC2Str(userLC, problemLC, timeLC, OnlyRight)
        if not os.path.exists(prefix + '_EERNN_input.csv'):
            self.FilterSubmitRecordData(userLC, problemLC, timeLC,OnlyRight)
        print('LoadSubmitRecordData ...')
        with open(prefix + '_EERNN_input.csv','r') as f:
            self.submitRecord.append([])
            for line in f:
                fields = line.strip().split(',')
                student, problem, is_correct = int(fields[0]), int(fields[1]), int(fields[2])
                while (student >= len(self.submitRecord)):
                    self.submitRecord.append([])
                self.submitRecord[student].append([problem, is_correct])
        with open(prefix + '_userName2userId.pkl', 'rb') as fr:
            self.userName2userId = pickle.load(fr)
        with open(prefix + '_userId2userName.pkl', 'rb') as fr:
            self.userId2userName = pickle.load(fr)
        with open(prefix + '_problemName2problemId.pkl', 'rb') as fr:
            self.problemName2problemId = pickle.load(fr)
        with open(prefix + '_problemId2problemName.pkl', 'rb') as fr:
            self.problemId2problemName = pickle.load(fr)


if __name__ == "__main__":
    pass
    # tmp = ojdataprocessor()
    # tmp.LoadSubmitRecordData([15,1000000,0.06,1],[10,1000000,0.02,1],['2005-01-01 23:47:31','2019-01-02 11:21:49'],OnlyRight=True)