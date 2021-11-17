# 心血管疾病（CVD）数据分析报告
## 0 分析框架  
<img src="https://user-images.githubusercontent.com/75609874/136686805-18480575-425d-4a94-a739-0510ee647852.png" width="100%">

## 1 分析背景和目的
### 1.1 分析背景
随着大数据应用范围的不断扩展，许多传统领域正在发生着巨大变革，而大数据应用到医疗健康领域后，一种惠及民生的新兴业态——”互联网+大健康”正式诞生。在“互联网+大健康”产业中，更加侧重的是尽可能客观而全面地了解病人的状况，在疾病尚未发生或者发生早期就对其进行预测。显然与传统医疗模式中等到发病时再“对症下药”相比，对病人进行早期治疗成本更低、治疗效果更加。
众所周知，心血管疾病近年来发病人数呈现增加趋势，也成为医疗界研究的重要课题。因此本文选取了一个心血管疾病的数据集，通过建模的方式对数据集进行探索，目的是利用患者的检查结果预测心血管疾病(CVD)的存在与否。
## 2 数据准备
### 2.1 数据来源
https://www.kaggle.com/sulianova/cardiovascular-disease-dataset#cardio_train.csv.   
数据集包括年龄、性别、收缩压、舒张压等12个特征的患者数据记录7万份。当患者有心血管疾病时，标签类“cardio”等于1，如果患者健康，则为0。所有数据集都是在体检时收集的。
### 2.2 数据描述
有三种类型的输入特征: Objective: 客观事实; Examination: 体检结果; Subjective: 患者提供的信息
| 特征 | 变量类型 | 变量 | Value Type |
| :----: | :------------: | :------: | :------: |
| Age | 客观事实 | 年龄 | Int(days)
| Height | 客观事实 | 身高 | Int(cm)
| Weight | 客观事实 | 年龄 | Float(kg)
| Gender | 客观事实 | 性别(1-F,2-M) | Categorical code
| Systolic blood pressure | 检查特征 | ap_hi（舒张压）| Int
| Diastolic blood pressure | 检查特征 | ap_lo（收缩压）| Int
| Cholesterol | 检查特征 | 胆固醇 |1：正常；2：高于正常；3、远高于正常
| Glucose| 检查特征 | gluc(葡萄糖)| 1：正常；2：高于正常；3、远高于正常
| Smoking| 主观特征 | smoke| binary(1:吸烟；0：不吸烟)
| Alcohol intake| 主观特征 | alco | binary(1:喝酒；0：不喝酒)
| Physical activity| 主观特征 | active| binary(1:参加；0：不参加)
| Presence or absence of cardiovascular disease| 目标变量 | Cardio| binary(1:患病；0：健康)  
### 2.3 创建环境分析环境并导入数据
```
import numpy as np
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns
#忽略警告
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
from sklearn.model_selection import train_test_split
#导入数据
df = pd.read_csv(r'/Users/Yuri Yuan/Desktop/cardio_train.csv',sep=';')
df.head()
```
![image](https://user-images.githubusercontent.com/75609874/142215413-ac34f062-6160-4498-811d-43df69767312.png)<br>
## 3 数据清洗<br>
### 3.1 查看数据信息
```
#查看数据信息
df.info()
```
<img src="https://user-images.githubusercontent.com/75609874/142217837-900f7c80-59fe-4d4e-bdf8-6dc4a50e4e7e.png" width="40%"><br>
数据加载后，数据基本情况如图。从图中可以看到加载后的数据一共70000行、13列，占用内存6.9MB。在数据类型上，所有特征均为数值型，12个整型和1个float类型，而且无缺失值。<br>
### 3.2 数据处理步骤<br>
<img src="https://user-images.githubusercontent.com/75609874/142218240-8b659a5e-e067-4ccb-83da-e454272e1069.png" width="100%">

```
#选择子集:id列无意义将其删除
df.drop(columns=['id'],inplace=True)
#选择子集:id列无意义将其删除
df.drop(columns=['id'],inplace=True)
#查看是否有重复值并删除
df.duplicated().sum()
df.drop_duplicates(keep='first',inplace =  True)
#查看是否存在缺失值
df.isnull().sum()
```
<img src="https://user-images.githubusercontent.com/75609874/142218970-9777dcca-8d53-43ec-b1f8-8e182a041002.png" width="20%">

```
#描述性统计分析
df.describe()  
```
<img src="https://user-images.githubusercontent.com/75609874/142219425-cd23e310-e58d-4a4b-becb-f9b6742b423c.png" width="120%"><br>
从表格中可以看出height、weight、ap_hi、ap_lo存在异常值,需要对异常值进行处理.仔细观察身高和体重,明显能注意到身高最小值55厘米,最大值高达250cm;体重最小值10kg,最大值200kg,这违背了人体自然规律;除此之外,血压是不能为负值的,且舒张压一般要低于收缩压。<br>
处理办法：体重和身高只选取选取2.5%-97.5%之间的数据（置信区间为95%）；删除舒张压大于收缩压的数据，然后依据正态分布原理，分别选取2.5%-97.5%的之间数据(置信区间95%)。<br>
```
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
```
<img src="https://user-images.githubusercontent.com/75609874/142222468-a3bb70e5-64bd-4f81-ab51-3a3f295ffb04.png" width="100%"><br>
## 4 理解数据
### 4.1 单变量分析
#### 1) 年龄与目标值之间的关系
```
from matplotlib import rcParams
#图像大小
rcParams['figure.figsize'] = 11,8
#将年龄天数(day)转化为年(year)
df['age'] = (df['age']/365).round().astype('int')
sns.countplot(x='age',hue='cardio',data=df,palette = 'Blues')
```
<img src="https://user-images.githubusercontent.com/75609874/142223853-2665a327-e968-40fa-9467-da2ed97e9f0b.png" width="100%"><br>
从图中可以观察到年龄55及以上的人群更容易患心血管疾病
#### 2) 身高体重与目标值之间的关系
```
#身高,体重与患病的关系
#图像大小
plt.figure(figsize=(12,8))
#子图1
plt.subplot(1,2,1)
sns.boxplot(x='cardio',y='height',data=df,palette='BuGn')
#子图2
plt.subplot(1,2,2)
sns.boxplot(x='cardio',y='weight',data=df,palette='Blues')
```
<img src="https://user-images.githubusercontent.com/75609874/142224135-7a72cd1e-d457-492b-8865-6d18f2e6c4a3.png" width="100%"><br>
#### 3) smoke、alco、active、gender与目标值之间的关系
```
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
```
<img src="https://user-images.githubusercontent.com/75609874/142224325-ef305040-9ab6-49cb-9804-e750a75a9494.png" width="100%"><br>
### 4.2 数据集中分类变量的分布情况双变量分析

```
df_categorical = df[['cholesterol','gluc','smoke','alco','active']]
data = pd.melt(df_categorical)
sns.countplot(x='variable',hue='value',data=data,palette='Blues')
```
<img src="https://user-images.githubusercontent.com/75609874/142225125-ae7bcd17-ac18-4170-b35f-51a1fff0c79d.png" width="100%"><br>
### 4.3 双变量分析
```
sns.catplot(x='variable',hue='value',col = 'cardio',data=pd.melt(df,id_vars=['cardio'],value_vars=['cholesterol','gluc','smoke','alco','active']),kind='count',palette='Blues')
```
<img src="https://user-images.githubusercontent.com/75609874/142225417-54f867e7-8a1b-4125-bd17-9aa4aead39ef.png" width="100%"><br>
从图中可以看出,CVD患者胆固醇和血糖水平较高,而且一般来说运动量较低.
### 4.4 创建一个新的特征:身体质量指数(BMI)= weight(kg)/height**2(m)
```
data = df[['gender','weight','height','alco','cardio']]
data['BMI'] = data['weight']/((data['height']/100)**2)
sns.catplot(x="gender", y="BMI", hue="alco", col="cardio", data=data, color = "yellow",kind="box", height=10, aspect=.7,palette='Blues')
```
<img src="https://user-images.githubusercontent.com/75609874/142226323-12fdf4f6-aac4-4e75-a772-73c7c17dcee2.png" width="100%"><br>
对比两幅图，根据女性的BMI，BMI越高，喝酒的女性比喝酒的男性患心血管疾病的风险更高。

## 5 模型构建和优化
### 5.0 导入机器学习相关算法包
```
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
```
### 5.1 多变量相关性分析
```
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
```
<img src="https://user-images.githubusercontent.com/75609874/142228290-4d61d427-0496-47ff-abb4-da35783f0b85.png" width="100%"><br>
### 5.2 模型构建
```
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
```
<img src="https://user-images.githubusercontent.com/75609874/142228857-26be5ae9-3f92-40f2-bcc4-5192b91628f4.png" width="40%"><br>

```
#数据标准化
scale = StandardScaler()
scale.fit(X_train)
X_train_scaled = scale.transform(X_train)
X_train_ = pd.DataFrame(X_train_scaled,columns=df.columns[:-1])
scale.fit(X_test)
X_test_scaled = scale.transform(X_test)
X_test_ = pd.DataFrame(X_test_scaled,columns=df.columns[:-1])
```
### 5.3 模型优化
```
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
```
<img src="https://user-images.githubusercontent.com/75609874/142229469-4344c171-404b-47c4-ba74-aee2464fde29.png" width="100%"><br>
<img src="https://user-images.githubusercontent.com/75609874/142229470-460c619e-76cf-4746-a93b-85b50192e674.png" width="100%"><br>
<img src="https://user-images.githubusercontent.com/75609874/142229490-8577a4a6-c591-40b1-912b-0ddf9ed4d6be.png" width="100%"><br>
<img src="https://user-images.githubusercontent.com/75609874/142229564-233a18a4-ac3e-4796-8536-c9f37b9d9c13.png" width="100%"><br>
<img src="https://user-images.githubusercontent.com/75609874/142229582-f12fd706-7511-4442-ac07-3f6ffabfb428.png" width="100%"><br>
<img src="https://user-images.githubusercontent.com/75609874/142230227-251b372f-d993-4edc-8a1c-4dc175e6f602.png" width="100%"><br>
从图中可以得出选择threshold在0.05时模型拟合的最好
### 5.4 模型选择
```
#final feature selection with threshold of 0.05
feat_final  = feat_select(0.05)
print(feat_final)
```
['age', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
#### 1) KNN表现
```
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
```
<img src="https://user-images.githubusercontent.com/75609874/142230289-75e94760-c7dd-4df8-b79e-03457001bbd1.png" width="100%"><br>

#### 2) 逻辑回归的表现
```
#logistic regression
lr.fit(X_train,y_train)
pred = lr.predict(X_val_)
#reports
print('Confusion Matrix = \n',confusion_matrix(y_val,pred))
print('\n',classification_report(y_val,pred))
```

![image](https://user-images.githubusercontent.com/75609874/142231285-25378ca4-9b38-4eb6-80ef-9f147fe9e3c7.png)

## 6 模型对比

```
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
        'score':[acc_random_forest,acc_svc,acc_linear_svc]}
)
models.sort_values(by='score',ascending = False)
```
![image](https://user-images.githubusercontent.com/75609874/142231604-929ecff9-e6fb-4d51-a441-5e76789da73f.png)<br>
从结果中可以看出随机森林的预测效果是较为不错，score达到97%；而其他模型score只有70%左右。



