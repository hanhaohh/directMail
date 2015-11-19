from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.mllib.regression import LabeledPoint
from numpy import array
from pyspark.mllib.linalg import Vectors
from pyspark.ml.classification import GBTClassifier,LogisticRegression
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import PCA
#init the sc
sc = SparkContext()
titile = sc.textFile("/user/demo/train.csv").take(1)[0]
sqlContext = SQLContext(sc)

#input 
rdd = sc.textFile("/user/demo/train.csv").filter(lambda x: x != titile).\
map(lambda x:x.split(","))
D = 2 ** 24 

def helper1(r):
    features=[]
    try:
        fe = r[1:-1]
        for i in range(len(fe)):
            features.append(float(abs(hash("VAR_"+'{0:04}'.format(i)+fe[i])))%D)
        target = float(r[-1])
        ID=float(r[0])
        return target, Vectors.dense(features)
    except:
        return (0.0,[0.0]*1932)
new_rdd = rdd.filter(lambda i : len(i)==1934)
rdd_after_trans = new_rdd.map(helper1)
rdd_after_trans.cache()
df = sqlContext.createDataFrame(rdd_after_trans,["label", "features"])
(trainingData, testData) = rdd_pca1.randomSplit([0.7, 0.3])

stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
si_model = stringIndexer.fit(trainingData)
td = si_model.transform(trainingData)

gbt = GBTClassifier(maxIter=100, maxDepth=10, labelCol="label")
model = gbt.fit(trainingData)
result = model.transform(testData).rdd.map(lambda r: str(r.label)+','+str(r.probability[0]))
result.saveAsTextFile("/user/demo/gbt_100_20")

df1 = sqlContext.createDataFrame([
     Row(label=0.0, features=Vectors.dense([0.0, 0.0])),
     Row(label=0.0, features=Vectors.dense([0.0, 1.0])),
     Row(label=1.0, features=Vectors.dense([1.0, 0.0]))])

