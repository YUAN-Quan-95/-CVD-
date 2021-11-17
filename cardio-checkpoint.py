#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns
#忽略警告
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
#逻辑回归算法
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#导入数据
df = pd.read_csv(r'D:\谷歌下载文件\cardio_train.csv',sep=';')
df.head()
#查看数据信息
df.info()
pandas_profiling.ProfileReport(df)

#数据清洗
#选择子集:id列无意义将其删除
df.drop(columns=['id'],inplace=True)
#查看是否有重复值并删除
df.duplicated().sum()
df.drop_duplicates(keep='first',inplace =  True)
#查看是否存在缺失值
df.isnull().sum()
#描述性统计分析
df.describe()
#异常值处理
#去除体重和身高在给定范围内低于或高于2.5%或97.5%
df.drop(df[(df['height'] > df['height'].quantile(0.975))|(df['height'] <df['height'].quantile(0.025))].index,inplace=True)
df.drop(df[(df['weight'] > df['weight'].quantile(0.975))|(df['weight'] <df['weight'].quantile(0.025))].index,inplace=True)
#查看有多少cases是舒张压大于收缩压
print('Diastilic pressure is higher than systolic one in {0} cases'.format(len(df[df['ap_lo']>df['ap_hi']])))
#去除舒张压大于收缩压的异常值
df.drop(df[df['ap_lo'] >df['ap_hi']].index,inplace=True)
#由于舒张压与收缩压极值存在问题，选取范围2.5%-97.5%
df.drop(df[(df['ap_lo'] > df['ap_lo'].quantile(0.975))|(df['ap_lo'] <df['ap_lo'].quantile(0.025))].index,inplace=True)
df.drop(df[(df['ap_hi'] > df['ap_hi'].quantile(0.975))|(df['ap_hi'] <df['ap_hi'].quantile(0.025))].index,inplace=True)
blood_pressure = df[['ap_lo','ap_hi']]
sns.boxplot(x='variable',y = 'value',data = blood_pressure.melt())

#理解数据
from matplotlib import rcParams
#图像大小
rcParams['figure.figsize'] = 11,8
#将年龄天数(day)转化为年(year)
df['age'] = (df['age']/365).round().astype('int')
sns.countplot(x='age',hue='cardio',data=df,palette = 'Blues')
#身高,体重与患病的关系
#图像大小
plt.figure(figsize=(12,8))
#子图1
plt.subplot(1,2,1)
sns.boxplot(x='cardio',y='height',data=df,palette='BuGn')
#子图2
plt.subplot(1,2,2)
sns.boxplot(x='cardio',y='weight',data=df,palette='Blues')

#子图1：smoke 与标签之间的关系
plt.figure(figsize=(8,4))
plt.subplot(1,3,1)
sns.barplot(data=df,x='smoke',y='cardio',palette='Blues')
#子图2：alco与标签之间的关系
plt.subplot(1,3,2)
sns.barplot(data=df,x='alco',y='cardio',palette='Accent_r')
#子图3：性别和患病的关系
plt.subplot(1,3,3)
sns.barplot(y='cardio',data=df,x='gender',palette='BuGn')


# 数据集中分类变量的分布情况
df_categorical = df[['cholesterol','gluc','smoke','alco','active']]
data = pd.melt(df_categorical)
sns.countplot(x='variable',hue='value',data=data,palette='Blues')

#双变量分析
sns.catplot(x='variable',hue='value',col = 'cardio',data=pd.melt(df,id_vars=['cardio'],value_vars=['cholesterol','gluc','smoke','alco','active']),kind='count',palette='Blues')

# 创建一个新的特征:身体质量指数(BMI)= weight(kg)/height**2(m)
data = df[['gender','weight','height','alco','cardio']]
data['BMI'] = data['weight']/((data['height']/100)**2)
sns.catplot(x="gender", y="BMI", hue="alco", col="cardio", data=data, color = "yellow",kind="box", height=10, aspect=.7,palette='Blues')

# 建模分析
#多变量相关性分析:各变量与cardio的相关系数
correlations = df.corr()
#创建具有亮值的连续调色板
cmap = sns.diverging_palette(220,10,as_cmap=True)
#下三角遮罩
mask =np.zeros_like(correlations,dtype=np.bool)
mask[np.triu_indices_from(mask)] =True
#图像设置
plt.style.use('seaborn-white')  
plt.rc('figure',figsize=(12,8),dpi=150)
#画出热力图,并校正长宽比
sns.heatmap(correlations,annot= True,fmt='.2f',cmap=cmap,mask=mask,vmax=.5,center=0,square=True,linewidths=.5,cbar_kws={"shrink":.5})
print(df.corr()['cardio'].drop('cardio'))

#编写预设函数:选取特征
def feat_select(threshold):
    abs_cor = df.corr()['cardio'].drop('cardio').abs()
    features = abs_cor[abs_cor>threshold].index.tolist()
    return features
def model(mod,X_train,X_test):
    mod.fit(X_train,y_train)
    pred = mod.predict(X_test)
    print('Model score = ',round(mod.score(X_test,y_test),4)*100,'%')

#切分数据集：随机将数据集切分为两部分
msk = np.random.rand(len(df))<0.85
df_train_test = df[msk]
df_val = df[~msk]

X = df_train_test.drop('cardio',axis=1)
y = df_train_test['cardio']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=70)

#逻辑回归
lr = LogisticRegression()
#训练模型
#设计6个阈值
threshold = [0.001,0.002,0.005,0.01,0.05,0.1]
for i in threshold:
    print('Threshold is {}'.format(i))
    feature_i = feat_select(i)
    X_train_i =X_train[feature_i]
    X_test_i = X_test[feature_i]
    model(lr,X_train_i,X_test_i)
#数据标准化
scale = StandardScaler()
scale.fit(X_train)
X_train_scaled = scale.transform(X_train)
X_train_ = pd.DataFrame(X_train_scaled,columns=df.columns[:-1])
scale.fit(X_test)
X_test_scaled = scale.transform(X_test)
X_test_ = pd.DataFrame(X_test_scaled,columns=df.columns[:-1])

#KNN优化
# optimum k with optimum threshold
for i in threshold:
    feature = feat_select(i)
    X_train_k = X_train_[feature]
    X_test_k = X_test_[feature]
    err = []
    for j in range(1,30):
        knn = KNeighborsClassifier(n_neighbors=j)
        knn.fit(X_train_k,y_train)
        pred_j = knn.predict(X_test_k)
        err.append(np.mean(y_test != pred_j))

    plt.figure(figsize=(10,6))
    plt.plot(range(1,30),err)
    plt.title('Threshold of {}'.format(i))
    plt.xlabel('K value')
    plt.ylabel('Error')
# 模型选择
#final feature selection with threshold of 0.05
feat_final  = feat_select(0.05)
print(feat_final)

#KNN表现
X_train = X_train_[feat_final]
X_val = np.asanyarray(df_val[feat_final])
y_val = np.asanyarray(df_val['cardio'])
scale.fit(X_val)
X_val_scaled = scale.transform(X_val)
X_val_ = pd.DataFrame(X_val_scaled,columns=df_val[feat_final].columns)
#knn with k = 15  
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train,y_train)
pred = knn.predict(X_val_)
print('Confusion Matrix =\n',confusion_matrix(y_val,pred))
print('\n',classification_report(y_val,pred))

#逻辑回归的表现
#logistic regression
lr.fit(X_train,y_train)
pred = lr.predict(X_val_)
#reports
print('Confusion Matrix = \n',confusion_matrix(y_val,pred))
print('\n',classification_report(y_val,pred))


#其他模型的表现
#随机森林
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(n_estimators=100)
#切分数据集
X = df_train_test.drop('cardio',axis=1)
y = df_train_test['cardio']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=70)
rf.fit(X_train, y_train)
Y_pred =rf.predict(X_test)
rf.score(X_train, y_train)
acc_random_forest = round(rf.score(X_train, y_train) * 100, 2)
acc_random_forest

from sklearn.svm import SVC, LinearSVC
svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 100, 2)
acc_svc

linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)
acc_linear_svc

models = pd.DataFrame({
        'model':['RandomForestClassifier','SVC','linearSVC'],
        'score':[acc_random_forest,acc_svc,acc_linear_svc]})
models.sort_values(by='score',ascending = False)





