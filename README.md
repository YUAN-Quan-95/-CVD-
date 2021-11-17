# 心血管疾病（CVD）数据分析报告
##0 分析框架  
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
| :------: | :------: | :------: | :------: |
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

