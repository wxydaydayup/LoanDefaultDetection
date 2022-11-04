# 阿里云-贷款违约预测

## 赛题说明

赛题链接：[阿里云-贷款违约预测](https://tianchi.aliyun.com/competition/entrance/531830/introduction)

代码仓库：

### 一、背景介绍 

赛题以金融风控中的个人信贷为背景，要求选手根据贷款申请人的数据信息预测其是否有违约的可能，以此判断是否通过此项贷款，这是一个典型的分类问题。

### 二、任务介绍

赛题以预测用户贷款是否违约为任务，数据集报名后可见并可下载，该数据来自某信贷平台的贷款记录，总数据量超过120w，包含47列变量信息，其中15列为匿名变量。为了保证比赛的公平性，将会从中抽取80万条作为训练集，20万条作为测试集A，20万条作为测试集B，同时会对employmentTitle、purpose、postCode和title等信息进行脱敏。

### 三、数据集

提交结果为每个测试样本是1的概率，也就是y为1的概率。评价方法为AUC评估模型效果（越大越好）。

提交前请确保预测结果的格式与sample_submit.csv中的格式一致，以及提交文件后缀名为csv。

形式如下：

```
id,isDefault
800000,0.5
800001,0.5
800002,0.5
800003,0.5
```

### 四、字段说明

|     **Field**      |                       **Description**                        |
| :----------------: | :----------------------------------------------------------: |
|         id         |                为贷款清单分配的唯一信用证标识                |
|      loanAmnt      |                           贷款金额                           |
|        term        |                       贷款期限（year）                       |
|    interestRate    |                           贷款利率                           |
|    installment     |                         分期付款金额                         |
|       grade        |                           贷款等级                           |
|      subGrade      |                        贷款等级之子级                        |
|  employmentTitle   |                           就业职称                           |
|  employmentLength  |                        就业年限（年）                        |
|   homeOwnership    |              借款人在登记时提供的房屋所有权状况              |
|    annualIncome    |                            年收入                            |
| verificationStatus |                           验证状态                           |
|     issueDate      |                        贷款发放的月份                        |
|      purpose       |               借款人在贷款申请时的贷款用途类别               |
|      postCode      |         借款人在贷款申请中提供的邮政编码的前3位数字          |
|     regionCode     |                           地区编码                           |
|        dti         |                          债务收入比                          |
| delinquency_2years |       借款人过去2年信用档案中逾期30天以上的违约事件数        |
|    ficoRangeLow    |            借款人在贷款发放时的fico所属的下限范围            |
|   ficoRangeHigh    |            借款人在贷款发放时的fico所属的上限范围            |
|      openAcc       |              借款人信用档案中未结信用额度的数量              |
|       pubRec       |                      贬损公共记录的数量                      |
| pubRecBankruptcies |                      公开记录清除的数量                      |
|      revolBal      |                       信贷周转余额合计                       |
|     revolUtil      | 循环额度利用率，或借款人使用的相对于所有可用循环信贷的信贷金额 |
|      totalAcc      |              借款人信用档案中当前的信用额度总数              |
| initialListStatus  |                      贷款的初始列表状态                      |
|  applicationType   |       表明贷款是个人申请还是与两个共同借款人的联合申请       |
| earliesCreditLine  |              借款人最早报告的信用额度开立的月份              |
|       title        |                     借款人提供的贷款名称                     |
|     policyCode     |      公开可用的策略_代码=1新产品不公开可用的策略_代码=2      |
|   n系列匿名特征    |        匿名特征n0-n14，为一些贷款人行为计数特征的处理        |

## BaseLine V1_lgb--分数: 0.7345

### 一、数据探索

#### 1、以直方图的形式展示数据信息

```python
import matplotlib.pyplot as plt
import seaborn as sns
# 以直方图的形式展示，isDefault表示显示数量
sns.countplot(x = 'grade',hue='isDefault',data=train)
```

#### 2、将数值类型的缺失值全部以中位数补全

```python
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

#数据加载
train=pd.read_csv("./train.csv")
testA=pd.read_csv("./testA.csv")
#将数值类型的缺失值全部以中位数补全
numerical_fea = list(train.select_dtypes(include=['float']).columns)
train[numerical_fea] = train[numerical_fea].fillna(train[numerical_fea].median())
test[numerical_fea] = test[numerical_fea].fillna(test[numerical_fea].median())
print('数值类型缺失值,中位数填充完成')
```

#### 3、将类别类型的缺失值全部以众数补全

```python
#将类别类型的缺失值全部以众数补全
from scipy import stats 
cat_fea = list(train.select_dtypes(exclude=['float']).columns)
for cf in cat_fea:
    if train[cf].isnull().sum():
        train[cf] = train[cf].fillna(stats.mode(train[cf])[0][0])
cat_fea = list(test.select_dtypes(exclude=['float']).columns)
for cf in cat_fea:
    if test[cf].isnull().sum():
        test[cf] = test[cf].fillna(stats.mode(test[cf])[0][0])
print('类别类型缺失值,众数填充完成')
```

### 二、特征工程

#### 1、时间多尺度变换与时间差计算

```python
import datetime
def create_days_diff(selected_cols):
    for selected in selected_cols:
        train[selected] = pd.to_datetime(train[selected])
        tmp_str = str(train[selected].min().year)+'-'+str("%02d" % train[selected].min().month)+'-'+str("%02d" % train[selected].min().day)
        startdate = datetime.datetime.strptime(tmp_str, '%Y-%m-%d')
        train[selected+'_diff'] = train[selected].apply(lambda x: x-startdate).dt.days
        
        test[selected] = pd.to_datetime(test[selected])
        tmp_str = str(test[selected].min().year)+'-'+str("%02d" % test[selected].min().month)+'-'+str("%02d" % test[selected].min().day)
        startdate = datetime.datetime.strptime(tmp_str, '%Y-%m-%d')
        test[selected+'_diff'] = test[selected].apply(lambda x: x-startdate).dt.days        
        print(selected+'_diff'+' 时间差字段 已经创建')
def create_ym_features(selected_cols):
    for selected in selected_cols:
        test[selected] = pd.to_datetime(test[selected])
        train_temp = pd.DatetimeIndex(train[selected])
        test_temp = pd.DatetimeIndex(test[selected])
        
        train[selected+'_year'] = train_temp.year
        test[selected+'_year'] = test_temp.year
        print(selected+'_year'+'字段 已经创建')
        
        train[selected+'_month'] = train_temp.month
        test[selected+'_month'] = test_temp.month
        print(selected+'_month'+'字段 已经创建')
        
selected_cols = ['issueDate','earliesCreditLine']
create_ym_features(selected_cols)
create_days_diff(selected_cols)
```

#### 2、将object字段编码

```python
for data in [train]:
    #贷款等级
    data['grade'] = data['grade'].map({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7})
    #就业年限（年）
    data['employmentLength'] = data['employmentLength'].map({'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5,'6 years':6,'7 years':7,'8 years':8,'9 years':9,'10+ years':10,'< 1 year':0})
    #贷款等级之子级
    data['subGrade'] = data['subGrade'].map({'E2':1,'D2':2,'D3':3,'A4':4,'C2':5,'A5':6,'C3':7,'B4':8,'B5':9,'E5':10,
        'D4':11,'B3':12,'B2':13,'D1':14,'E1':15,'C5':16,'C1':17,'A2':18,'A3':19,'B1':20,
        'E3':21,'F1':22,'C4':23,'A1':24,'D5':25,'F2':26,'E4':27,'F3':28,'G2':29,'F5':30,
        'G3':31,'G1':32,'F4':33,'G4':34,'G5':35})    
    #借款人信用档案中当前的信用额度总数 除以 贷款金额
    data['rato']=data['totalAcc']/data['loanAmnt']
for data in [test]:
    #贷款等级
    data['grade'] = data['grade'].map({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7})
    #就业年限（年）
    data['employmentLength'] = data['employmentLength'].map({'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5,'6 years':6,'7 years':7,'8 years':8,'9 years':9,'10+ years':10,'< 1 year':0})
    #贷款等级之子级
    data['subGrade'] = data['subGrade'].map({'E2':1,'D2':2,'D3':3,'A4':4,'C2':5,'A5':6,'C3':7,'B4':8,'B5':9,'E5':10,
        'D4':11,'B3':12,'B2':13,'D1':14,'E1':15,'C5':16,'C1':17,'A2':18,'A3':19,'B1':20,
        'E3':21,'F1':22,'C4':23,'A1':24,'D5':25,'F2':26,'E4':27,'F3':28,'G2':29,'F5':30,
        'G3':31,'G1':32,'F4':33,'G4':34,'G5':35})    
    #借款人信用档案中当前的信用额度总数 除以 贷款金额
    data['rato']=data['totalAcc']/data['loanAmnt']
```

### 三、模型建立

使用CatBoost模型

```python
#CatBoost模型
model=model = CatBoostClassifier(
    loss_function="Logloss",    # 分类任务常用损失函数
    eval_metric="Accuracy",     # 表示用于过度拟合检测和最佳模型选择的度量标准；
    learning_rate=0.08,         # 表示学习率
    iterations=10000,
    random_seed=42,           # 设置随机种子进行固定
    od_type="Iter",
    metric_period=20,           # 与交叉验证folds数匹配
    max_depth = 8,              # 表示树模型最大深度
    early_stopping_rounds=500,  # 早停步数
    use_best_model=True,
    task_type="GPU",          # 数据量较小，GPU加速效果不明显
    bagging_temperature=0.9,
    leaf_estimation_method="Newton",
)
n_folds =10 #十折交叉校验
answers = []
mean_score = 0
data_x=train.drop(['isDefault'],axis=1)
data_y=train[['isDefault']].copy()
sk = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2021)
all_test = test.copy()
for train, test in sk.split(data_x, data_y):  
    x_train = data_x.iloc[train]
    y_train = data_y.iloc[train]
    x_test = data_x.iloc[test]
    y_test = data_y.iloc[test]
    clf = model.fit(x_train,y_train, eval_set=(x_test,y_test),verbose=500) # 500条打印一条日志
    
    yy_pred_valid=clf.predict(x_test,prediction_type='Probability')[:,-1]
    print('cat验证的auc:{}'.format(roc_auc_score(y_test, yy_pred_valid)))
    mean_score += roc_auc_score(y_test, yy_pred_valid) / n_folds
    
    y_pred_valid = clf.predict(all_test,prediction_type='Probability')[:,-1]
    answers.append(y_pred_valid) 
print('mean valAuc:{}'.format(mean_score))
cat_pre=sum(answers)/n_folds
result['isDefault']=cat_pre
result.to_csv('./baselinev1.csv',index=False)
```



## BaseLine V1_lgb--分数: 0.7349

### 一、特征工程优化

1、**min，max，mean，std没有意义，因为是单一特征**

**只增加count，isDefault_mean（唯一值个数少的情况，如果唯一值多，会早成标签泄露问题）**

```python
#cat_fea 为类别特征（一般是出float以外，时间类型也不算）
cat_fea =['term','grade','subGrade','employmentLength','homeOwnership','verificationStatus','purpose','regionCode','initialListStatus','applicationType']
df = pd.concat([train,test],axis = 0)
for col in cat_fea:
    temp = train.groupby(col,as_index=False)[col].agg({col+'_count': 'count'})
    df = pd.merge(df,temp,on = col,how = 'left')
    #isDefault为标签项，需要替换
    temp = train.groupby(col,as_index=False)['isDefault'].agg({col+'_isDefault_mean': 'mean'})
    df = pd.merge(df,temp,on = col,how = 'left')
train = df[df['isDefault'].notnull()]
test = df[df['isDefault'].isnull()]
```

