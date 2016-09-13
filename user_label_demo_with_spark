# coding: utf-8
#!/usr/bin/python

"""
2016.09.02 create by SparkUser
文件功能描述：用户画像案例演示（利用Spark sql模块建立一张用户标签表）
"""

# 导入需要的package
from pyspark.sql.types import *
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext,Row
import pandas as pd

"""
2016.09.02 create by SparkUser
类功能描述: 获取数据文件；数据清洗和格式转化；聚类分析；结果输出
"""
class SparkCreateTable(object):

    # P2P用户数据存放路径
    file_path = 'YourFilePath'

    def spark_create_table_main(self):
        # 创建一个SparkContext对象，用来配置客户端，告诉Spark如何访问集群、本例以local模式运行
        conf = SparkConf().setAppName('Spark').setMaster('local')
        sc = SparkContext(conf=conf)

        print('模块一：读取数据文件存入RDD中')

        # 读取P2P用户数据文件，存入RDD（Spark中的基本抽象-弹性分布式数据集）
        rdd = sc.textFile(self.file_path)
        sql_context = SQLContext(sc)

        # 输出RDD总元素个数
        print('RDD中总元素个数：%s' % rdd.count())

        print('模块二：数据预处理')
        # 将RDD文件中的元素按行分割开
        rdd = rdd.map(lambda line: line.split(","))

        # 输出RDD文件前5行
        print('RDD前五行：')
        print(pd.DataFrame(rdd.take(5)))

        # 删除header行
        header = rdd.first()
        rdd = rdd.filter(lambda line:line != header)

        # 将RDD文件转成Spark DataFrame
        df = rdd.map(lambda line: Row(invest_cust_id = line[0], borr_cust_id = line[1], borr_amt=line[2], total_ret=line[3],first_ret_date=line[4],lend_date=line[5],last_ret_date=line[6],max_ret_amt=line[7],min_ret_amt=line[8],ret_cnt=line[9])).toDF()

        # 查看DataFrame
        print('Spark DataFrame前5行:')
        df.show(5)

        # 将DataFrame存入一张临时表，命名为Temp
        df.registerTempTable("Temp")

        print('模块三：调用PySpark.sql模块创建结果表并查看结果集')
        # 利用PySpark.sql模块创建结果表、在原始表基础上根据一定规则，创建一个新列，标注每笔P2P标的的还款方式
        print('根据SQL建立新表，请等待...')
        sql='''select *,'一次还本付息'as type from Temp where ret_cnt=1 
                    union select *,'先息后本' as type from Temp where ret_cnt>1 and max_ret_amt>=borr_amt
                    union select *,'等额本息' as type from Temp where ret_cnt>1 and max_ret_amt<borr_amt 
                        and (max_ret_amt-total_ret/ret_cnt)/max_ret_amt<0.05
                        and (max_ret_amt-min_ret_amt)/max_ret_amt<0.05
                    union select *,'等额本金' as type from Temp where  ret_cnt>1 and max_ret_amt<borr_amt 
                        and (max_ret_amt-total_ret/ret_cnt)/max_ret_amt<0.05
                        and (max_ret_amt-min_ret_amt)/max_ret_amt>=0.05
                    union select *,'其他' as type from Temp where ret_cnt>1 and max_ret_amt<borr_amt 
                        and (max_ret_amt-total_ret/ret_cnt)/max_ret_amt>=0.05 '''
        df_userlabel=sql_context.sql(sqlQuery=sql)
        

        # 查看结果表前5行
        print('建表完成，输出结果表前5行：')
        df_userlabel.show(5)

        # 数据筛选功能演示
        print('筛选放款日期为2015年3月6日的记录子集：')
        df_userlabel.filter(df_userlabel.lend_date=='20150306').show(5)

        # 重命名某列演示
        print("将type列重命名为'还款方式' ")
        df_userlabel.withColumnRenamed('type','还款方式').show(5)

        # 转化成Pandas DataFrame格式
        print("将Spark DataFrame转化成Pandas DataFrame")
        print(df_userlabel.toPandas().head(10))

        sc.stop()

if __name__ == '__main__':
    
    print('Spark 服务开始')
    SparkCreateTable = SparkCreateTable()
    SparkCreateTable.spark_create_table_main()
    print('Spark 服务结束')

