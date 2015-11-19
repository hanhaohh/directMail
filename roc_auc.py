
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
fw = open("rf_100_10.csv","r")
data = fw.readlines()
label = map(lambda i: float(i.split(",")[0].strip()),data)
score = map(lambda i: float(i.split(",")[1].strip()),data)
fpr, tpr, thresholds = metrics.roc_curve(label,score, pos_label=0)
roc_auc = auc(fpr, tpr)


#LR_PCA_500_001
fw1 = open("rf_100_15.csv","r")
data1 = fw1.readlines()
label1 = map(lambda i: int(float(i.split(",")[0].strip())),data1)
score1 = map(lambda i: float(i.split(",")[1].strip()),data1)
fpr1, tpr1, thresholds1 = metrics.roc_curve(label1, score1, pos_label=0)
roc_auc1 = auc(fpr1, tpr1)
print roc_auc,roc_auc1
# # #LR_PCA_1000_001
fw2 = open("rf_100_20.csv","r")
data2 = fw2.readlines()
label2 = map(lambda i: int(float(i.split(",")[0].strip())),data2)
score2 = map(lambda i: float(i.split(",")[1].strip()),data2)
fpr2, tpr2, thresholds2 = metrics.roc_curve(label2, score2, pos_label=0)
roc_auc2 = auc(fpr2, tpr2)

plt.figure()

plt.plot(fpr, tpr, label='Random Forest with 100 tree: ROC curve (area = %0.3f)' % roc_auc)
plt.plot(fpr1, tpr1, label='Random Forest with 200 tree: ROC curve (area = %0.3f)' % roc_auc1)
plt.plot(fpr2, tpr2, label='Random Forest with 500 tree: ROC curve (area = %0.3f)' % roc_auc2)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])


plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()