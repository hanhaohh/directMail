from pyspark import SparkConf,SparkContext
import numpy
import pyfiles
import logi_sgd
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import Vectors
from pyspark.sql import Row
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.types import *
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel

train='/user/demo/train.csv'
test='test.csv'
submission = 'sgd_subm.csv'  # path of to be outputted submission file

conf = SparkConf().setAppName("logi_sgd")
sc = SparkContext(conf=conf,pyFiles= pyfiles.common_python_files)
sqlContext = SQLContext(sc)
data = sc.textFile(train)
title = data.take(1)[0]
def func(r):
	r=r.asDict()
	x = []
	label =''
	ID=''
	target=0.0
	try:
		for i in [str(i) for i in title.split(",")]:
			if i == "target" :
				target = float(r["target"])
			if i =="ID" :
				ID = r.ID			
			else:
				x.append(float(abs(hash(i + '_' + r[i]))) % D)
	except:
		target=0.0
		features =[0.0]*1934
	return Row(target=target,features=x)

# B, model
alpha = .005  	# learning rate
beta = 1		
L1 = 0.     	# L1 regularization, larger value means more regularized
L2 = 0.     	# L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 24             # number of weights to use
interaction = False     # whether to enable poly2 feature interactions

# D, training/validation
epoch = 1       # learn training data for N passes
holdafter = None   # data after date N (exclusive) are used as validation
holdout = 100  # use every N training instance for holdout validation
def data_gen(input_data,D):
	learner = logi_sgd.gradient_descent(alpha, beta, L1, L2, D, interaction)
	rdd = sc.textFile("/user/demo/train.csv").filter(lambda x: x != title).filter(lambda x: len(x)!=1934).map(lambda x:x.split(","))
	all_types = []
	for i in [str(i) for i in title.split(",")]:
		schema = all_types.append( StructField(i, StringType(), True) )
	schema = StructType(all_types)
	df = sqlContext.createDataFrame(rdd,schema)
	#one-hot encode everything with hash trick
	df_new = df.rdd.map(lambda x: func(x))
	return df_new
def tran(i):
	return LabeledPoint(i.target,i.features)
rdd= data_gen(train,D).map(tran)

# print type(rdd.take(1))
# print (rdd.take(1)[0])
model = LinearRegressionWithSGD.train(rdd)
#model = LogisticRegressionWithLBFGS.train(rdd)
##############################################################################
# start training #############################################################
##############################################################################


# initialize ourselves a learner

# start training
print('Training Learning started; total 150k training samples')

# for e in range(epoch):
#     loss = 0.
#     count = 0
#     for t,  x, y in data_gen(train, D):  # data is a generator
#         p = learner.predict(x)
#         loss += logloss(p, y)
#         print x
#         learner.update(x, p, y)
#         count+=1
#         if count%15000==0:
#             print('%s\tencountered: %d\tcurrent logloss: %f' % (datetime.now(), count, loss/count))

#import pickle
#pickle.dump(learner,open('sgd_adapted_learning.p','w'))

##############################################################################
# start testing, and build Kaggle's submission file ##########################
##############################################################################
# count=0
# print('Testing started; total 150k test samples')
# with open(submission, 'w') as outfile:
#     outfile.write('ID,target\n')
#     for  ID, x, y in data(test, D):
#         count+=1
#         if count%15000==0:
#             print('%s\tencountered: %d' % (datetime.now(), count))
#         p = learner.predict(x)
#         outfile.write('%s,%s\n' % (ID, str(p)))
