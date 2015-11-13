from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
#Connect to a specific Spark/ Master
sc = SparkContext()
rdd = sc.textFile("wasb://han@hanhaohh.blob.core.windows.net/train.csv").filter(lambda x: x != title).\
map(lambda x:x.split(","))
all_types = []
for i in [str(i) for i in title.split(",")]:
    schema = all_types.append( StructField(i, StringType(), True) )
    schema = StructType(all_types)
from pyspark.sql import Row
from pyspark.mllib.classification import LogisticRegressionWithSGD
from numpy import array
from pyspark.mllib.regression import LabeledPoint
D = 2 ** 24 
def helper1(r):
    features=[]
    try:
        fe = r[1:-1]
        for i in range(len(fe)):
            features.append(float(abs(hash("VAR_"+str(i)+fe[i])))%D)
        target = float(r[-1])
        ID=float(r[0])
        return LabeledPoint(target,features)
    except:
        return LabeledPoint(0.0,[0.0]*1932)
new_rdd = rdd.filter(lambda i : len(i)==1934)
df = new_rdd.map(helper1)

model = LogisticRegressionWithSGD.train(df)
df.take(1)