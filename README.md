# 心血管疾病（CVD）数据分析报告
## 0 分析框架  
![image](https://user-images.githubusercontent.com/75609874/136686805-18480575-425d-4a94-a739-0510ee647852.png)
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
#逻辑回归算法
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
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
数据加载后，数据基本情况如图。从图中可以看到加载后的数据一共70000行、13列，占用内存6.9MB。在数据类型上，所有特征均为数值型，12个整型和1个float类型，而且无缺失值。  
<img src="https://user-images.githubusercontent.com/75609874/142217837-900f7c80-59fe-4d4e-bdf8-6dc4a50e4e7e.png" width="40%">
### 3.2 数据处理步骤<br>
![image](https://user-images.githubusercontent.com/75609874/142218240-8b659a5e-e067-4ccb-83da-e454272e1069.png)

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
<img src="https://user-images.githubusercontent.com/75609874/142218970-9777dcca-8d53-43ec-b1f8-8e182a041002.png" width="30%">

```
#描述性统计分析
df.describe()
```
<img src="https://user-images.githubusercontent.com/75609874/142219425-cd23e310-e58d-4a4b-becb-f9b6742b423c.png" width="100%">

从表格中可以看出height、weight、ap_hi、ap_lo存在异常值,需要对异常值进行处理.仔细观察身高和体重,明显能注意到身高最小值55厘米,最大值高达250cm;体重最小值10kg,最大值200kg,这违背了人体自然规律;除此之外,血压是不能为负值的,且舒张压一般要低于收缩压.

