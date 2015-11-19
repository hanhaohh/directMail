import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn import metrics
#LR_PCA_100_001
def calculate(file_name,mode,threshold):
	if mode == "prediction":
		fw = open(file_name,"r")
		data = fw.readlines()
		label = map(lambda i: float(i.split(",")[0].strip()),data)
		score = map(lambda i: float(i.split(",")[1].strip()),data)
		labelandScore = zip(label,score) 
		tp = 0
		tn = 0
		fp =  0
		fn=0
		for i,j in labelandScore:
			if i==j==0:
				tn=tn+1
			if i==j==1:
				tp=tp+1
			if (i==0 and j==1):
				fp = fp+1
			if (i ==1 and j==0):
				fn = fn+1
		accuracy =(tp+tn)/float(tp+tn+fp+fn)
		return accuracy
	else:
		fw = open(file_name,"r")
		data = fw.readlines()
		label = map(lambda i: float(i.split(",")[0].strip()),data)
		score = map(lambda i: float(i.split(",")[1].strip()),data)
		labelandScore = zip(label,score) 
		tp = 0
		tn = 0
		fp =  0
		fn=0
		for i,j in labelandScore:
			if (i==0.0) and (j>threshold):
				tn=tn+1
			if (i==1.0) and (j<threshold):
				tp=tp+1
			if (i==0.0) and (j<threshold):
				fp = fp+1
			if (i ==1.0) and (j>threshold):
				fn = fn+1
		accuracy =(tp+tn)/float(tp+tn+fp+fn)
		return accuracy
	print accuracy



a=[]
# for i in range(1,100,1): 
# 	a.append(calculate("result_lr_pca_500_001_02.csv","other",i/100.0)) 
print calculate("gbt_100.csv",mode="prediction",threshold=1)