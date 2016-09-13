# coding: utf-8
#!/usr/bin/python


"""
2016.09.02 create by SparkUser
文件功能描述：精准营销案例演示
"""

# 导入需要的package
from pyspark.sql.types import *
from pyspark.sql import Row
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.feature import StandardScaler
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import matplotlib.pyplot as plt
import pandas as pd

"""
2016.09.02 create by SparkUser
类功能描述: 获取数据文件；数据清洗和格式转化；聚类分析；结果输出
"""
class SparkClusterAnalysis(object):

    # P2P用户数据存放路径
    file_path = 'YourFilePath'

    def spark_cluster_analysis_main(self):

        # 创建一个SparkContext对象，用来配置客户端，告诉Spark如何访问集群、本例以local模式运行
        conf = SparkConf().setAppName('Spark').setMaster('local')
        sc = SparkContext(conf=conf)
        
        print('模块一：读取数据文件存入RDD中')
        # 读取P2P用户数据文件，存入RDD（Spark中的基本抽象-弹性分布式数据集）
        rdd=sc.textFile(self.file_path)
        sql_context = SQLContext(sc)

        # 输出RDD总元素个数
        print('RDD中总元素个数：%s' % rdd.count())

        print('模块二：数据预处理')
        # 将RDD文件中的元素按行分割开
        rdd = rdd.map(lambda line: line.split(","))

        # 输出RDD文件前5行
        print('RDD前五行：')
        print(pd.DataFrame(rdd.take(5)))

        # 去除header行
        print('删除header')
        header = rdd.first()
        rdd = rdd.filter(lambda line:line != header)

        # 输出RDD文件前5行
        print('RDD前五行：')
        print(pd.DataFrame(rdd.take(5)))

        # 将RDD文件转成Spark DataFrame，以便后续分析
        df = rdd.map(lambda line: Row(_id = line[0], recency = line[1], bid_num=line[2],avg_bid_amt=line[3])).toDF()
        
        
        # 筛选聚类变量
        print('删除id列')
        features = df.map(lambda i: i[1:])

        # 输出RDD文件前5行
        print('RDD前五行：')
        print(pd.DataFrame(features.take(5)))

        # 进行数据标准化操作：计算每个变量值的z标准分数
        standardizer = StandardScaler(withMean=True,withStd=True)

        # 将数据转换为稠密向量格式
        features_transform = standardizer.fit(features).transform(features)
        print('输出标准化后向量格式的数据：')
        print(features_transform.take(5))
        
        print('模块三：使用Pyspark.mllib机器学习库进行聚类分析')
        print('聚类分析运行中，请等待...')
        
        # 训练k=2-10共9个聚类模型，根据Within-Cluster Sum of Square类内平方和来选择最佳聚类个数
        result=[]
        for k in range(2,11):
            model=KMeans.train(features_transform,k,maxIterations=30, runs=3, initializationMode="random")
            cost=model.computeCost(features_transform)
            result.append(cost)

        # 输出每个聚类模型的类内距离平方和，评估数据最好模型
        print('输出每个聚类模型的类内距离平方和，选择效果最佳模型')
        for k in range(2,11):
            print('Within-Cluster Sum of Square for k=%d is %d'%(k,result[k-2]))

        # 输出聚类模型碎石图，帮助选择最佳模型
        get_ipython().magic('matplotlib inline')
        plt.title('Scree-plots for Cluster Models')
        plt.plot([2,3,4,5,6,7,8,9,10],result,'r*-')

        # 训练最佳模型并分类
        model=KMeans.train(features_transform,7,maxIterations=30, runs=3, initializationMode="random")
        prediction=model.predict(features_transform).collect()
        

        print('模块四：结果输出')
        #将每个样本的所属类赋给原始数据集
        final_result=pd.concat([df.toPandas(),pd.DataFrame(prediction)],axis=1)
        final_result.rename(columns={0:'cluster'},inplace=True)
        # 转变列格式为float型
        final_result['recency']=final_result['recency'].astype(float)
        final_result['bid_num']=final_result['bid_num'].astype(float)
        final_result['avg_bid_amt']=final_result['avg_bid_amt'].astype(float)

        # 输出聚类后每个群组在3个指标上的平均值，从而区分出各类的特征
        print('聚类结果前5行')
        print(final_result.head())
        print('输出每个类在3个指标上的平均值，从而区分各类特征')
        print(final_result.groupby('cluster').mean())
        sc.stop()
        


if __name__ == '__main__':
    print('Spark 服务开始')
    SparkClusterAnalysis = SparkClusterAnalysis()
    SparkClusterAnalysis.spark_cluster_analysis_main()
    print('Spark 服务结束')

