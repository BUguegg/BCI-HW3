import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy
import math
from scipy.spatial.distance import cdist

class Neural_Spike_Classification():
    def __init__(self):
        #load Data
        data_path = './data/neural_Data.mat'
        neural_data = scipy.io.loadmat(data_path)

        self.chan_names = np.array(neural_data['chan_names'])
        self.chan = [name[0][0] for name in self.chan_names]
        self.cursor_pos = np.array(neural_data['cursor_pos'])
        self.finger_pos = np.array(neural_data['finger_pos'])
        self.spikes = np.array(neural_data['spikes'])
        self.t = np.array(neural_data['t'])
        self.target_pos = np.array(neural_data['target_pos'])
        self.wf = np.array(neural_data['wf'])

        # print("chan_names.shape:",self.chan_names.shape)
        # print("cursor_pos.shape:",self.cursor_pos.shape)
        # print("finger_pos.shape:",self.finger_pos.shape)
        # print("spikes.shape:",self.spikes.shape)
        # print("t.shape:",self.t.shape)
        # print("target_pos.shape:",self.target_pos.shape)
        # print("wf.shape:",self.wf.shape)

        self.resultPath = './result/Task1'
        pass
    
    #Slice Data according to self.target_pos
    def sliceMotor(self):
        moterStateSlice = []
        t_pos = [self.target_pos[0]]   #start 0
        tSlice = [self.t[0][0]]       #start 0
        # tSlice = []
        comx, comy = self.target_pos[0]
        sit = 0
        for idx, co in enumerate(self.target_pos):
            x = co[0]
            y = co[1]         
            if x == comx and y == comy :
                continue
            else:
                # print(idx)
                comx = x
                comy = y
                t_pos.append(co)
                moterStateSlice.append(self.cursor_pos[sit:idx]) 
                tSlice.append(self.t[idx][0])
                sit = idx
                # break
        # print(len(moterStateSlice))
        # print(len(tSlice))
        # print(len(t_pos))
        # print(t_pos[-1])
        return moterStateSlice, tSlice, t_pos

    #Slice Spikes according to self.t
    def sliceSpike(self, channel, u, tSlice):
        SpikeSlice = []
        wfSlice = []
        start = 0
        t = 0
        for id, spike in enumerate(self.spikes[channel][u]):
            if spike >= tSlice[t]:
                SpikeSlice.append(self.spikes[channel][u][start:id])
                wfSlice.append(self.wf[channel][u][start:id])
                start = id
                if t == len(tSlice) - 1:
                    SpikeSlice.append(self.spikes[channel][u][id:-1])
                    wfSlice.append(self.wf[channel][u][start:id])
                    break
                else:
                    t += 1
            else:
                continue

        head = SpikeSlice.pop(0)
        tail = SpikeSlice.pop(-1)
        wfSlice.pop(0)
        wfSlice.pop(-1)
        # print(len(SpikeSlice))
        # print(len(head))
        # print(len(tail))
        return  SpikeSlice, wfSlice

    def Classification(self, moterStateSlice, t_pos, SpikeSlice, wfSlice):
        SpkiceClass = [[] for i in range(8)]
        MotorClass = [[] for i in range(8)]
        wfClass = [[] for i in range(8)]
        # print(np.array(t_pos).shape)
        x = np.array(t_pos).T[0]
        y = np.array(t_pos).T[1]

        for i in range(1, len(x)):
            angle = math.atan2((y[i] - y[i-1]) , ( x[i] - x[i-1]))
            angle = 180 - int(angle * 180 / math.pi)
            if angle >= 0 and angle < 45:
                SpkiceClass[0].append(SpikeSlice[i-1])
                MotorClass[0].append(moterStateSlice[i-1])
                wfClass[0].append(wfSlice[i-1])
            elif angle >= 45 and angle < 90:
                SpkiceClass[1].append(SpikeSlice[i-1])
                MotorClass[1].append(moterStateSlice[i-1])
                wfClass[1].append(wfSlice[i-1])
            elif angle >= 90 and angle < 135:
                SpkiceClass[2].append(SpikeSlice[i-1])
                MotorClass[2].append(moterStateSlice[i-1])
                wfClass[2].append(wfSlice[i-1])
            elif angle >= 135 and angle < 180:
                SpkiceClass[3].append(SpikeSlice[i-1])
                MotorClass[3].append(moterStateSlice[i-1])
                wfClass[3].append(wfSlice[i-1])
            elif angle >= 180 and angle < 225:
                SpkiceClass[4].append(SpikeSlice[i-1])
                MotorClass[4].append(moterStateSlice[i-1])
                wfClass[4].append(wfSlice[i-1])
            elif angle >= 225 and angle < 270:
                SpkiceClass[5].append(SpikeSlice[i-1])
                MotorClass[5].append(moterStateSlice[i-1])
                wfClass[5].append(wfSlice[i-1])
            elif angle >= 270 and angle < 315:
                SpkiceClass[6].append(SpikeSlice[i-1])
                MotorClass[6].append(moterStateSlice[i-1])
                wfClass[6].append(wfSlice[i-1])
            elif angle >= 315 and angle < 360:
                SpkiceClass[7].append(SpikeSlice[i-1])
                MotorClass[7].append(moterStateSlice[i-1])
                wfClass[7].append(wfSlice[i-1])
        
        # for j in range(len(SpkiceClass)):
        #     print(len(SpkiceClass[j]))
        #     print(len(MotorClass[j]))
        #     print("###################")

        return SpkiceClass, MotorClass, wfClass


    #PCA
    def pca(self):
        # print(self.wf.shape)
        # print(self.wf[2][0].shape)
        # fig = plt.figure()
        # plt.plot(self.wf[2][0])
        # plt.show()
        # fig.savefig("F:\\Course\\BCI_HW2\\BCI_hw3\\results\\orginalWF.png")


        pca = PCA(n_components=3)
        pca.fit(self.wf[2][0])
        # print(pca.explained_variance_ratio_)
        # print(pca.explained_variance_)

        wf_new = pca.transform(self.wf[2][0])
        # fig = plt.figure()
        # plt.plot(wf_new)
        # plt.show()
        # fig.savefig("F:\\Course\\BCI_HW2\\BCI_hw3\\results\\PCA_to_3.png")

        moterStateSlice, tSlice, t_pos = self.sliceMotor()
        SpikeSlice, wfSlice = self.sliceSpike(channel=2, u=0, tSlice=tSlice)
        SpikeClass, MotorClass, wfClass =  self.Classification(moterStateSlice, t_pos, SpikeSlice, wfSlice)
        
        wf_pca = [[] for i in range(8)]
        label = []
        # for wf in wfSlice:
        #     wf_pca.append(pca.transform(wf))
        for dir in range(len(wfClass)):
            for wf in wfClass[dir]:
                wf_pca[dir].append(pca.transform(wf))
                label.append(dir)

        # count = 0
        # for pca in wf_pca:
        #     print(len(pca))
        #     count += len(pca)
        
        # print(count)
        # print(len(label))

        return wf_pca, wf_new, label

    #K-means
    def K_means(self):
        wf_pca, wf_new, label = self.pca()
        Data = []
        for k_data in wf_pca:
            Data += k_data
        # print(len(Data))
        # print(np.array(Data).shape)
        # print(np.array(data).shape)

        #wf_new
        #生成一个字典保存每次的代价函数
        distortions = []
        K = range(1,10)
        for k in K:
            #分别构建各种K值下的聚类器
            Model = KMeans(n_clusters=k).fit(wf_new) 
            #计算各个样本到其所在簇类中心欧式距离(保存到各簇类中心的距离的最小值)
            distortions.append(sum(np.min(cdist(wf_new, Model.cluster_centers_, 'euclidean'), axis=1)) / wf_new.shape[0])

        #绘制各个K值对应的簇内平方总和，即代价函数SSE
        #可以看出当K=3时，出现了“肘部”，即最佳的K值。
        # plt.plot(K,distortions,'bx-')
        # #设置坐标名称
        # plt.xlabel('optimal K')
        # plt.ylabel('SSE')
        # # plt.show()
        # plt.savefig("F:\\Course\\BCI_HW2\\BCI_hw3\\results\\Different_Kernel_K-means.png")

        model = KMeans(n_clusters=8) #构造聚类器
        model.fit(wf_new) #拟合我们的聚类模型
        label_pred = model.labels_ #获取聚类标签
        ctr = model.cluster_centers_  #获取聚类中心
        inertia = model.inertia_ #获取SSE
        print("计算得到聚类平方误差总和为",inertia)        

        #绘制K-Means结果
        #取出每个簇的样本
        x0 = wf_new[label_pred == 0]
        x1 = wf_new[label_pred == 1]
        x2 = wf_new[label_pred == 2]
        x3 = wf_new[label_pred == 3]
        x4 = wf_new[label_pred == 4]
        x5 = wf_new[label_pred == 5]
        x6 = wf_new[label_pred == 6]
        x7 = wf_new[label_pred == 7]
        #分别绘出各个簇的样本
        fig = plt.figure()
        plt.scatter(x0, x0, 
                    c = "red", marker='o', label='label0')
        plt.scatter(x1, x1, 
                    c = "green", marker='*', label='label1')
        plt.scatter(x2, x2, 
                    c = "blue", marker='+', label='label2')
        plt.scatter(x3, x3, 
                    c = "pink", marker='^', label='label3')
        plt.scatter(x4, x4, 
                    c = "grey", marker='>', label='label4')
        plt.scatter(x5, x5, 
                    c = "yellow", marker='<', label='label3')
        plt.scatter(x6, x6, 
                    c = "orange", marker='|', label='label4')
        plt.scatter(x7, x7, 
                    c = "purple", marker='d', label='label3')
        plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],
                    c = "black", marker='s',label='centroids')
        plt.title('Sum of squared error:'+str(inertia))
        # plt.ylabel('sepal width')
        plt.legend(loc=2) 
        plt.show()
        fig.savefig("F:\\Course\\BCI_HW2\\BCI_hw3\\results\\K-means8.png")



if __name__ == "__main__":
    SpikeClass = Neural_Spike_Classification()
    # SpikeClass.pca()
    SpikeClass.K_means()
