
![png](images/house-0713/image.png)

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

    /kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv
    /kaggle/input/house-prices-advanced-regression-techniques/data_description.txt
    /kaggle/input/house-prices-advanced-regression-techniques/train.csv
    /kaggle/input/house-prices-advanced-regression-techniques/test.csv
    

## STEP1. 필수라이브러리 불러오기


```python
import pandas as pd 
import numpy as np 
import matplotlib as mpl 
import seaborn as sns 
import sklearn
import xgboost as xgb 
import lightgbm as lgb

print("pandas version :", pd.__version__)
print("numpy version :", np.__version__)
print("matplotlib version :", mpl.__version__)
print("seaborn version :", sns.__version__)
print("scikit-learn version :", sklearn.__version__)
print("xgboost version :", xgb.__version__)
print("lightgbm version :", lgb.__version__)
```


<style type='text/css'>
.datatable table.frame { margin-bottom: 0; }
.datatable table.frame thead { border-bottom: none; }
.datatable table.frame tr.coltypes td {  color: #FFFFFF;  line-height: 6px;  padding: 0 0.5em;}
.datatable .bool    { background: #DDDD99; }
.datatable .object  { background: #565656; }
.datatable .int     { background: #5D9E5D; }
.datatable .float   { background: #4040CC; }
.datatable .str     { background: #CC4040; }
.datatable .time    { background: #40CC40; }
.datatable .row_index {  background: var(--jp-border-color3);  border-right: 1px solid var(--jp-border-color0);  color: var(--jp-ui-font-color3);  font-size: 9px;}
.datatable .frame tbody td { text-align: left; }
.datatable .frame tr.coltypes .row_index {  background: var(--jp-border-color0);}
.datatable th:nth-child(2) { padding-left: 12px; }
.datatable .hellipsis {  color: var(--jp-cell-editor-border-color);}
.datatable .vellipsis {  background: var(--jp-layout-color0);  color: var(--jp-cell-editor-border-color);}
.datatable .na {  color: var(--jp-cell-editor-border-color);  font-size: 80%;}
.datatable .sp {  opacity: 0.25;}
.datatable .footer { font-size: 9px; }
.datatable .frame_dimensions {  background: var(--jp-border-color3);  border-top: 1px solid var(--jp-border-color0);  color: var(--jp-ui-font-color3);  display: inline-block;  opacity: 0.6;  padding: 1px 10px 1px 5px;}
</style>



    pandas version : 1.3.5
    numpy version : 1.21.6
    matplotlib version : 3.5.2
    seaborn version : 0.11.2
    scikit-learn version : 1.0.2
    xgboost version : 1.6.1
    lightgbm version : 3.3.2
    

## STEP2.데이터 불러오기


```python
DATA_PATH = '/kaggle/input/house-prices-advanced-regression-techniques/'
train = pd.read_csv(DATA_PATH + "train.csv")
test = pd.read_csv(DATA_PATH + "test.csv")
submission = pd.read_csv(DATA_PATH + 'sample_submission.csv')
print("데이터 불러오기 완료!")
```

    데이터 불러오기 완료!
    

## STEP3.데이터 둘러보기
- 데이터 변수 파악
- 데이터 전처리


```python
print(train.shape, test.shape, submission.shape)
```

    (1460, 81) (1459, 80) (1459, 2)
    


```python
print(train.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1460 non-null   int64  
     1   MSSubClass     1460 non-null   int64  
     2   MSZoning       1460 non-null   object 
     3   LotFrontage    1201 non-null   float64
     4   LotArea        1460 non-null   int64  
     5   Street         1460 non-null   object 
     6   Alley          91 non-null     object 
     7   LotShape       1460 non-null   object 
     8   LandContour    1460 non-null   object 
     9   Utilities      1460 non-null   object 
     10  LotConfig      1460 non-null   object 
     11  LandSlope      1460 non-null   object 
     12  Neighborhood   1460 non-null   object 
     13  Condition1     1460 non-null   object 
     14  Condition2     1460 non-null   object 
     15  BldgType       1460 non-null   object 
     16  HouseStyle     1460 non-null   object 
     17  OverallQual    1460 non-null   int64  
     18  OverallCond    1460 non-null   int64  
     19  YearBuilt      1460 non-null   int64  
     20  YearRemodAdd   1460 non-null   int64  
     21  RoofStyle      1460 non-null   object 
     22  RoofMatl       1460 non-null   object 
     23  Exterior1st    1460 non-null   object 
     24  Exterior2nd    1460 non-null   object 
     25  MasVnrType     1452 non-null   object 
     26  MasVnrArea     1452 non-null   float64
     27  ExterQual      1460 non-null   object 
     28  ExterCond      1460 non-null   object 
     29  Foundation     1460 non-null   object 
     30  BsmtQual       1423 non-null   object 
     31  BsmtCond       1423 non-null   object 
     32  BsmtExposure   1422 non-null   object 
     33  BsmtFinType1   1423 non-null   object 
     34  BsmtFinSF1     1460 non-null   int64  
     35  BsmtFinType2   1422 non-null   object 
     36  BsmtFinSF2     1460 non-null   int64  
     37  BsmtUnfSF      1460 non-null   int64  
     38  TotalBsmtSF    1460 non-null   int64  
     39  Heating        1460 non-null   object 
     40  HeatingQC      1460 non-null   object 
     41  CentralAir     1460 non-null   object 
     42  Electrical     1459 non-null   object 
     43  1stFlrSF       1460 non-null   int64  
     44  2ndFlrSF       1460 non-null   int64  
     45  LowQualFinSF   1460 non-null   int64  
     46  GrLivArea      1460 non-null   int64  
     47  BsmtFullBath   1460 non-null   int64  
     48  BsmtHalfBath   1460 non-null   int64  
     49  FullBath       1460 non-null   int64  
     50  HalfBath       1460 non-null   int64  
     51  BedroomAbvGr   1460 non-null   int64  
     52  KitchenAbvGr   1460 non-null   int64  
     53  KitchenQual    1460 non-null   object 
     54  TotRmsAbvGrd   1460 non-null   int64  
     55  Functional     1460 non-null   object 
     56  Fireplaces     1460 non-null   int64  
     57  FireplaceQu    770 non-null    object 
     58  GarageType     1379 non-null   object 
     59  GarageYrBlt    1379 non-null   float64
     60  GarageFinish   1379 non-null   object 
     61  GarageCars     1460 non-null   int64  
     62  GarageArea     1460 non-null   int64  
     63  GarageQual     1379 non-null   object 
     64  GarageCond     1379 non-null   object 
     65  PavedDrive     1460 non-null   object 
     66  WoodDeckSF     1460 non-null   int64  
     67  OpenPorchSF    1460 non-null   int64  
     68  EnclosedPorch  1460 non-null   int64  
     69  3SsnPorch      1460 non-null   int64  
     70  ScreenPorch    1460 non-null   int64  
     71  PoolArea       1460 non-null   int64  
     72  PoolQC         7 non-null      object 
     73  Fence          281 non-null    object 
     74  MiscFeature    54 non-null     object 
     75  MiscVal        1460 non-null   int64  
     76  MoSold         1460 non-null   int64  
     77  YrSold         1460 non-null   int64  
     78  SaleType       1460 non-null   object 
     79  SaleCondition  1460 non-null   object 
     80  SalePrice      1460 non-null   int64  
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB
    None
    


```python
print(test.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1459 entries, 0 to 1458
    Data columns (total 80 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1459 non-null   int64  
     1   MSSubClass     1459 non-null   int64  
     2   MSZoning       1455 non-null   object 
     3   LotFrontage    1232 non-null   float64
     4   LotArea        1459 non-null   int64  
     5   Street         1459 non-null   object 
     6   Alley          107 non-null    object 
     7   LotShape       1459 non-null   object 
     8   LandContour    1459 non-null   object 
     9   Utilities      1457 non-null   object 
     10  LotConfig      1459 non-null   object 
     11  LandSlope      1459 non-null   object 
     12  Neighborhood   1459 non-null   object 
     13  Condition1     1459 non-null   object 
     14  Condition2     1459 non-null   object 
     15  BldgType       1459 non-null   object 
     16  HouseStyle     1459 non-null   object 
     17  OverallQual    1459 non-null   int64  
     18  OverallCond    1459 non-null   int64  
     19  YearBuilt      1459 non-null   int64  
     20  YearRemodAdd   1459 non-null   int64  
     21  RoofStyle      1459 non-null   object 
     22  RoofMatl       1459 non-null   object 
     23  Exterior1st    1458 non-null   object 
     24  Exterior2nd    1458 non-null   object 
     25  MasVnrType     1443 non-null   object 
     26  MasVnrArea     1444 non-null   float64
     27  ExterQual      1459 non-null   object 
     28  ExterCond      1459 non-null   object 
     29  Foundation     1459 non-null   object 
     30  BsmtQual       1415 non-null   object 
     31  BsmtCond       1414 non-null   object 
     32  BsmtExposure   1415 non-null   object 
     33  BsmtFinType1   1417 non-null   object 
     34  BsmtFinSF1     1458 non-null   float64
     35  BsmtFinType2   1417 non-null   object 
     36  BsmtFinSF2     1458 non-null   float64
     37  BsmtUnfSF      1458 non-null   float64
     38  TotalBsmtSF    1458 non-null   float64
     39  Heating        1459 non-null   object 
     40  HeatingQC      1459 non-null   object 
     41  CentralAir     1459 non-null   object 
     42  Electrical     1459 non-null   object 
     43  1stFlrSF       1459 non-null   int64  
     44  2ndFlrSF       1459 non-null   int64  
     45  LowQualFinSF   1459 non-null   int64  
     46  GrLivArea      1459 non-null   int64  
     47  BsmtFullBath   1457 non-null   float64
     48  BsmtHalfBath   1457 non-null   float64
     49  FullBath       1459 non-null   int64  
     50  HalfBath       1459 non-null   int64  
     51  BedroomAbvGr   1459 non-null   int64  
     52  KitchenAbvGr   1459 non-null   int64  
     53  KitchenQual    1458 non-null   object 
     54  TotRmsAbvGrd   1459 non-null   int64  
     55  Functional     1457 non-null   object 
     56  Fireplaces     1459 non-null   int64  
     57  FireplaceQu    729 non-null    object 
     58  GarageType     1383 non-null   object 
     59  GarageYrBlt    1381 non-null   float64
     60  GarageFinish   1381 non-null   object 
     61  GarageCars     1458 non-null   float64
     62  GarageArea     1458 non-null   float64
     63  GarageQual     1381 non-null   object 
     64  GarageCond     1381 non-null   object 
     65  PavedDrive     1459 non-null   object 
     66  WoodDeckSF     1459 non-null   int64  
     67  OpenPorchSF    1459 non-null   int64  
     68  EnclosedPorch  1459 non-null   int64  
     69  3SsnPorch      1459 non-null   int64  
     70  ScreenPorch    1459 non-null   int64  
     71  PoolArea       1459 non-null   int64  
     72  PoolQC         3 non-null      object 
     73  Fence          290 non-null    object 
     74  MiscFeature    51 non-null     object 
     75  MiscVal        1459 non-null   int64  
     76  MoSold         1459 non-null   int64  
     77  YrSold         1459 non-null   int64  
     78  SaleType       1458 non-null   object 
     79  SaleCondition  1459 non-null   object 
    dtypes: float64(11), int64(26), object(43)
    memory usage: 912.0+ KB
    None
    


```python
print(submission.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1459 entries, 0 to 1458
    Data columns (total 2 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   Id         1459 non-null   int64  
     1   SalePrice  1459 non-null   float64
    dtypes: float64(1), int64(1)
    memory usage: 22.9 KB
    None
    

### 데이터 변수(칼럼) 살펴보기
1. SalePrice 집가격
2. OverallQual 집재료 및 마감평가
3. GrLivArea 지상 거실 면적
4. GarageArea 차고 크기
5. TotalBsmtSF 지하 총 면적
6. 1stFlrSF 1층 면적
7. FullBath 전체 욕실
8. TotRmsAbvGrd 총 실수(욕실 미포함)
9. YearBuilt 건축년도
10. YearRemodAdd 리모델링년도
11. Fireplaces 벽난로 수
12. BsmtFinSF1 
13. LotFrontage
14. 2ndFlrSF
15. HalfBath
16. LotArea
17. BsmtFullBath
18. BedroomAbvG
19. BsmtFinBath
20. BsmtHalfBath
21. LowQualfinSF
22. OverallCond
23. MSSubClass
24. KitchenAbvGr

- ->데이터 변수가 79개나 된다. 제거해도 될만한 컬럼을 제거해보자.


### 데이터 변수(칼럼) 살펴보기
1. SalePrice 집가격
2. OverallQual 집재료 및 마감평가
3. GrLivArea 지상 거실 면적
4. GarageArea 차고 크기
5. GarageCars 차고 대수
6. TotalBsmtSF 1층(기본층) 총 면적
7. 1stFlrSF 2층 면적
8. FullBath 전체 욕실들(욕조,샤워,변기,세면대) 
9. TotRmsAbvGrd 총 실수(욕실 미포함)
10. YearBuilt 건축년도
11. YearRemodAdd 리모델링 년도
12. GarageYrBlt 차고만든 년도
13. MasVnrArea 석조표면마무리 면적
14. Fireplaces 벽난로 수
15. BsmtFinSF1 지층 1차마감 면의 질 
16. LotFrontage 집과 연결된 길까지의 거리
17. 2ndFlrSF 3층 면적
18. HalfBath 반쪽 욕실들(양변기,세면대)
19. OpenPorchSF 열린 현관 면적
20. LotArea 대지면적
21. BsmtFullBath 지층 풀 욕실
22. BedroomAbvG 지층포함 위쪽 침실
23. ScreenPorch 그물 현관 면적
24. PoolArea 수영장면적
25. MoSold 팔린 달
26. BsmtFinsF2 지층 2차마감 면의 질
27. BsmtHalfBath 지층의 반쪽 면적
28. LowQualfinSF 전층에서 낮은 마감의 면
29. OverallCond 전체적인 조건 평가
30. MSSubClass 건물특성별 점수
31. EnclosedPorch 막힌 현관 면적
32. KitchenAbvGr 부엌 수


```python
# 상관분석
train.corrwith(train['SalePrice']).sort_values(ascending=False)
```




    SalePrice        1.000000
    OverallQual      0.790982
    GrLivArea        0.708624
    GarageCars       0.640409
    GarageArea       0.623431
    TotalBsmtSF      0.613581
    1stFlrSF         0.605852
    FullBath         0.560664
    TotRmsAbvGrd     0.533723
    YearBuilt        0.522897
    YearRemodAdd     0.507101
    GarageYrBlt      0.486362
    MasVnrArea       0.477493
    Fireplaces       0.466929
    BsmtFinSF1       0.386420
    LotFrontage      0.351799
    WoodDeckSF       0.324413
    2ndFlrSF         0.319334
    OpenPorchSF      0.315856
    HalfBath         0.284108
    LotArea          0.263843
    BsmtFullBath     0.227122
    BsmtUnfSF        0.214479
    BedroomAbvGr     0.168213
    ScreenPorch      0.111447
    PoolArea         0.092404
    MoSold           0.046432
    3SsnPorch        0.044584
    BsmtFinSF2      -0.011378
    BsmtHalfBath    -0.016844
    MiscVal         -0.021190
    Id              -0.021917
    LowQualFinSF    -0.025606
    YrSold          -0.028923
    OverallCond     -0.077856
    MSSubClass      -0.084284
    EnclosedPorch   -0.128578
    KitchenAbvGr    -0.135907
    dtype: float64



- 강한 상관치가 보이는 데이터는 OverallQual, GarageCars, GarageArea, TotalBsmtSF, 1stFlrSF, FullBath, TotRmsAbvGrd,  YearBuilt, YearRemodAdd를 우선적으로 선택한다.     
- 결측치가 보이는 데이터는 보완  생략을 한다.

## STEP4. 탐색적 자료분석
- 시각화
  + 타겟변수의 상태 확인
  + 변수와 종속변수(SalePrice)의 시각화

### 1.타겟변수(SalePricce)를 확인하여 시각화 해본다.


```python
import matplotlib.pyplot as plt 
from scipy.stats import norm 
(mu, sigma) = norm.fit(train['SalePrice'])
print("평균:", mu)
print("표준편차:", sigma)

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(train['SalePrice'])
ax.set(title="SalePrice Distribution")
ax.axvline(mu, color = 'r', linestyle = '--')
ax.text(mu + 10000, 160, 'Mean of SalePrice', color = 'r')
plt.show()
```

    평균: 180921.19589041095
    표준편차: 79415.29188606751
    


    
![png](images/house-0713/output_17_1.png)
    


### 2. 변수와 종속변수(SalePrice)의 시각화
- 상관계수가 0.4이상의 히트맵 그래프를 그려본다.


```python
corrmat = train.corr()
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>=0.4]
#히트맵
plt.figure(figsize=(13,10))
sns.heatmap(train[top_corr_features].corr(),annot=True)
```




    <AxesSubplot:>




    
![png](images/house-0713/output_19_1.png)
    


- OverallQual, GarageCars와 GarageArea(0.88), GrLiveArea와 TotRmsAbvGrd(0.82), TotalBsmtSF와 1stFlrSF(0.82), GarageYtBuilk와 YearBuilt(0.83),FullBath, YearRemodAdd인 것을 보았을 때 
- OverallQual(0.79)>GrLiveArea(0.71)>GarageCars(0.64)> TotalBsmtSF(0.61)>YearBulit(0.52)>FullBath(0.56)>YearRemodAdd(0.51)만을 사용한다.

- 이중 TotalBsmtSF,GarageCars는 결측치가 있기에 채우도록 하자.
- 집재료 및 마감평가>지상 거실 면적> 차고대수 > 지층 총면적 > 건축연도 > 전체욕실들 > 리모델링 연도 


```python
fig, ((ax1, ax2), (ax3, ax4),(ax5,ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(16,13)) 
# SP - OverallQual
OverallQual_scatter_plot = pd.concat([train['SalePrice'],train['OverallQual']],axis = 1) 
sns.regplot(x='OverallQual',y = 'SalePrice',data = OverallQual_scatter_plot,scatter= True, fit_reg=True, ax=ax1) 

# SP - TotalBsmtSF
TotalBsmtSF_scatter_plot = pd.concat([train['SalePrice'],train['TotalBsmtSF']],axis = 1) 
sns.regplot(x='TotalBsmtSF',y = 'SalePrice',data = TotalBsmtSF_scatter_plot,scatter= True, fit_reg=True, ax=ax2) 

# SP - GrLivArea
GrLivArea_scatter_plot = pd.concat([train['SalePrice'],train['GrLivArea']],axis = 1) 
sns.regplot(x='GrLivArea',y = 'SalePrice',data = GrLivArea_scatter_plot,scatter= True, fit_reg=True, ax=ax3) 

# SP - GarageCars
GarageCars_scatter_plot = pd.concat([train['SalePrice'],train['GarageCars']],axis = 1) 
sns.regplot(x='GarageCars',y = 'SalePrice',data = GarageCars_scatter_plot,scatter= True, fit_reg=True, ax=ax4) 

# SP - FullBath
FullBath_scatter_plot = pd.concat([train['SalePrice'],train['FullBath']],axis = 1) 
sns.regplot(x='FullBath',y = 'SalePrice',data = FullBath_scatter_plot,scatter= True, fit_reg=True, ax=ax5) 

# SP - YearBuilt
YearBuilt_scatter_plot = pd.concat([train['SalePrice'],train['YearBuilt']],axis = 1) 
sns.regplot(x='YearBuilt',y = 'SalePrice',data = YearBuilt_scatter_plot,scatter= True, fit_reg=True, ax=ax6) 

# SP - YearRemodAdd
YearRemodAdd_scatter_plot = pd.concat([train['SalePrice'],train['YearRemodAdd']],axis = 1) 
YearRemodAdd_scatter_plot.plot.scatter('YearRemodAdd','SalePrice')

```




    <AxesSubplot:xlabel='YearRemodAdd', ylabel='SalePrice'>




    
![png](images/house-0713/output_21_1.png)
    



    
![png](images/house-0713/output_21_2.png)
    


- OverallQual, GarageCars, FullBath는 범주형의 형태


```python
fig, ax = plt.subplots(nrows = 1, ncols = 3)
fig.tight_layout()
fig.set_size_inches(15, 5)

sns.boxplot(x = 'OverallQual', y = 'SalePrice', data = train, ax=ax[0])
sns.boxplot(x = 'GarageCars',y = 'SalePrice', data = train, ax=ax[1])
sns.boxplot(x = 'FullBath', y = 'SalePrice', data = train, ax=ax[2])

ax[0].set_title("OverallQualr")
ax[1].set_title("GarageCars")
ax[2].set_title("FullBath")

plt.show()
```


    
![png](images/house-0713/output_23_0.png)
    



```python
##### 범주형 변수와 가격사이의 비교
figure, ((ax1,ax2,ax3,ax4), (ax5,ax6,ax7,ax8)) = plt.subplots(nrows=2, ncols=4)
figure.set_size_inches(18,8)

sns.barplot(data=train, x="MSSubClass", y="SalePrice", ax=ax1)
sns.barplot(data=train, x="LotShape", y="SalePrice", ax=ax2)
sns.barplot(data=train, x="BldgType", y="SalePrice", ax=ax3)
sns.barplot(data=train, x="HouseStyle", y="SalePrice", ax=ax4)
sns.barplot(data=train, x="KitchenAbvGr", y="SalePrice", ax=ax5)
sns.barplot(data=train, x="Functional", y="SalePrice", ax=ax6)
sns.barplot(data=train, x="SaleType", y="SalePrice", ax=ax7)
sns.barplot(data=train, x="SaleCondition", y="SalePrice", ax=ax8)

#ax1.set(ylabel='주거타입',title="연도별 가격")
# ax2.set(xlabel='부지모양',title="월별 가격")
# ax3.set(xlabel='주거 타입', title="일별 가격")
# ax4.set(xlabel='주거 스타일', title="시간별 가격")
# ax4.set(xlabel='최초공사년도', title="시간별 가격")
# ax4.set(xlabel='리모델링년도', title="시간별 가격")
```




    <AxesSubplot:xlabel='SaleCondition', ylabel='SalePrice'>




    
![png](images/house-0713/output_24_1.png)
    


## STEP5. 데이터 전처리
-  


```python
# 데이터 ID값 제거
train_ID = train['Id']
test_ID = test['Id']

train_t = train.drop(['Id'], axis = 1)
test_t = test.drop(['Id'], axis = 1)
print(train_t.shape,test_t.shape)
```

    (1460, 80) (1459, 79)
    


```python
#로그 안된 세일가격포함 데이터 합치기
all_ds = pd.concat([train_t, test_t]).reset_index(drop=True)
all_ds.shape
```




    (2919, 80)




```python
GrLivArea_scatter_plot = pd.concat([train['SalePrice'],train['GrLivArea']],axis = 1) 
sns.regplot(x='GrLivArea',y = 'SalePrice',data = GrLivArea_scatter_plot,scatter= True, fit_reg=True)
```




    <AxesSubplot:xlabel='GrLivArea', ylabel='SalePrice'>




    
![png](images/house-0713/output_28_1.png)
    


- 왼쪽 2개는 이상치인데 제거해야하나 말아야 하나....


```python
# 로그변환을 함. 
train['SalePrice'] = np.log1p(train['SalePrice'])

(mu, sigma) = norm.fit(train['SalePrice'])
print("평균:", mu)
print("표준편차:", sigma)

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(train['SalePrice'])
ax.set(title="SalePrice Distribution")
ax.axvline(mu, color = 'r', linestyle = '--')
ax.text(mu + 0.0001, 160, 'Mean of SalePrice', color = 'r')
ax.set_ylim(0, 170)
plt.show()
```

    평균: 12.024057394918406
    표준편차: 0.39931245219387496
    


    
![png](images/house-0713/output_30_1.png)
    



```python
# y값 추출 -> train 데이터에서 saleprice만 따로 저장
y = train['SalePrice']
train_t = train_t.drop('SalePrice', axis = 1)
print(train_t.shape)
```

    (1460, 79)
    


```python
# 데이터 합치기
all_df = pd.concat([train_t, test_t]).reset_index(drop=True)
all_df.shape
```




    (2919, 79)




```python
# 결측치 확인
def check_na(data, head_num = 6):
  isnull_na = (data.isnull().sum() / len(data)) * 100
  data_na = isnull_na.drop(isnull_na[isnull_na == 0].index).sort_values(ascending=False)
  missing_data = pd.DataFrame({'Missing Ratio' :data_na, 
                               'Data Type': data.dtypes[data_na.index]})
  print("결측치 데이터 컬럼과 건수:\n", missing_data.head(head_num))

check_na(all_df, 20)
```

    결측치 데이터 컬럼과 건수:
                   Missing Ratio Data Type
    PoolQC            99.657417    object
    MiscFeature       96.402878    object
    Alley             93.216855    object
    Fence             80.438506    object
    FireplaceQu       48.646797    object
    LotFrontage       16.649538   float64
    GarageFinish       5.447071    object
    GarageQual         5.447071    object
    GarageCond         5.447071    object
    GarageYrBlt        5.447071   float64
    GarageType         5.378554    object
    BsmtExposure       2.809181    object
    BsmtCond           2.809181    object
    BsmtQual           2.774923    object
    BsmtFinType2       2.740665    object
    BsmtFinType1       2.706406    object
    MasVnrType         0.822199    object
    MasVnrArea         0.787941   float64
    MSZoning           0.137033    object
    BsmtFullBath       0.068517   float64
    


```python
# 필요한 데이터 컬럼만 모음
all_df1 = all_df[['OverallQual','GrLivArea','GarageCars',
                  'TotalBsmtSF','YearBuilt','FullBath','YearRemodAdd']]
print(all_df1.shape)
check_na(all_df1, 20)
```

    (2919, 7)
    결측치 데이터 컬럼과 건수:
                  Missing Ratio Data Type
    GarageCars        0.034258   float64
    TotalBsmtSF       0.034258   float64
    


```python
#import numpy as np
# 이번에는 수치형 데이터만 추출한다. 
num_all_vars = list(all_df1.select_dtypes(include=[np.number]))
print("The whole number of all_vars", len(num_all_vars))

# 이번에는 수치형 데이터의 평균이 아닌 중간값을 지정했다. 
for i in num_all_vars:
  # all_df1[i].fillna(value=all_df1[i].median(), inplace=True)
    print(i)
    all_df1[i] = all_df1[i].fillna(value=all_df1[i].median())  
check_na(all_df1, 20)
```

    The whole number of all_vars 7
    OverallQual
    GrLivArea
    GarageCars
    TotalBsmtSF
    YearBuilt
    FullBath
    YearRemodAdd
    결측치 데이터 컬럼과 건수:
     Empty DataFrame
    Columns: [Missing Ratio, Data Type]
    Index: []
    

    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      # Remove the CWD from sys.path while we load stuff.
    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      # Remove the CWD from sys.path while we load stuff.
    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      # Remove the CWD from sys.path while we load stuff.
    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      # Remove the CWD from sys.path while we load stuff.
    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      # Remove the CWD from sys.path while we load stuff.
    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      # Remove the CWD from sys.path while we load stuff.
    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      # Remove the CWD from sys.path while we load stuff.
    


```python
all_df1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2919 entries, 0 to 2918
    Data columns (total 7 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   OverallQual   2919 non-null   int64  
     1   GrLivArea     2919 non-null   int64  
     2   GarageCars    2919 non-null   float64
     3   TotalBsmtSF   2919 non-null   float64
     4   YearBuilt     2919 non-null   int64  
     5   FullBath      2919 non-null   int64  
     6   YearRemodAdd  2919 non-null   int64  
    dtypes: float64(2), int64(5)
    memory usage: 159.8 KB
    


```python
cate_name = ['OverallQual','GrLivArea','YearBuilt','FullBath','YearRemodAdd']
for c in cate_name:
    all_df1[c] =all_df[c].astype('category')

all_df1.dtypes
```

    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    




    OverallQual     category
    GrLivArea       category
    GarageCars       float64
    TotalBsmtSF      float64
    YearBuilt       category
    FullBath        category
    YearRemodAdd    category
    dtype: object




```python
#One-hot Encoding
#all_df1 = pd.get_dummies(all_df).reset_index(drop=True)
#all_df1.shape
```


```python
#train,test데이터 재분리
X = all_df1.iloc[:len(y),:]
test1 = all_df1.iloc[len(y):,:] 

X.shape, y.shape, test1.shape

```




    ((1460, 7), (1460,), (1459, 7))




```python
print(y)#->로그변환 확인
```

    0       12.247699
    1       12.109016
    2       12.317171
    3       11.849405
    4       12.429220
              ...    
    1455    12.072547
    1456    12.254868
    1457    12.493133
    1458    11.864469
    1459    11.901590
    Name: SalePrice, Length: 1460, dtype: float64
    

## STEP 6.모델링


```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
```


```python
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
from sklearn.preprocessing import MinMaxScaler
```


```python
# 선형회귀모델
lr_reg = LinearRegression()
# 피처에 대한 표준화 진행과 K-fold를 함께 진행
pipe = make_pipeline(MinMaxScaler(),lr_reg)
scores = cross_validate(pipe,X,y,cv=5,
                        scoring='neg_mean_squared_error',return_train_score =True)
print("MSLE: {0:.3f}".format(np.mean(-scores['test_score'])))
```

    MSLE: 0.030
    


```python
#rf랜덤포레스트
np.random.seed(0)
rf = RandomForestRegressor(n_estimators = 300)
# 피처에 대한 표준화 진행과 K-fold를 함께 진행
pipe = make_pipeline(MinMaxScaler(),rf)
scores = cross_validate(pipe,X,y,cv=5,
                        scoring='neg_mean_squared_error',return_train_score =True)
print("MSLE: {0:.3f}".format(np.mean(-scores['test_score'])))
```

    MSLE: 0.026
    


```python
#LGBM
lgbm = LGBMRegressor(n_estimators = 500, objective = 'regression')
# 피처에 대한 표준화 진행과 K-fold를 함께 진행
pipe = make_pipeline(MinMaxScaler(),lgbm)
scores = cross_validate(pipe,X,y,cv=5,
                        scoring='neg_mean_squared_error',return_train_score =True)
print("MSLE: {0:.3f}".format(np.mean(-scores['test_score'])))
```

    MSLE: 0.031
    


```python
# 하이퍼 파라미터 튜닝
#pipeline = Pipeline([('scaler',MinMaxScaler()),('lgbm',LGBMRegressor(objective='regression'))])
#params={'lgbm__learnin_rate':[0.001,0.01,0.1],
#       'lgbm__max_depth':[5,10],
#       'lgbm__reg_lambda':[0.1,1],
#       'lgbm__subsample':[0.5,1],
#       'lgbm__n_estimators':[500,1000]}
#grid_model = GridSearchCV(pipeline,param_grid=params,scoring='neg_mean_squared_error',
#                         cv=5, n_jobs=5,verbose=True)
#grid_model.fit(X,y)
#print("MSLE:{0:.3f}".format(-1*grid_model.best_score_))
#print('optimal hyperparameter:',grid_model.best_params_)
```


```python
# 피쳐 표준화
minmax = MinMaxScaler()
minmax.fit(X) # 훈련셋 모수 분포 저장
X_scaled = minmax.transform(X)
X_test_scaled = minmax.transform(test1)

# 최종파라미터 튜닝된 모델로 학습
lgbm = LGBMRegressor(n_estimators = 500, objective = 'regression',
                     learning_rate = 0.06, max_depth = 3, reg_lambda =1 ,subsample = 0.52, 
                     random_state = 7)

# 학습
lgbm.fit(X_scaled, y)
```




    LGBMRegressor(learning_rate=0.06, max_depth=3, n_estimators=500,
                  objective='regression', random_state=7, reg_lambda=1,
                  subsample=0.52)




```python
# test1에 대한 예측
pred = lgbm.predict(X_test_scaled)
fpred = np.expm1(pred)# 로그변환을 풀어줌

#lgbm 모델의 feature importance
imp = pd.DataFrame({'feature': test1.columns,
                   'coefficient':lgbm.feature_importances_})
imp = imp.sort_values(by = 'coefficient', ascending = False)

plt.barh(imp['feature'],imp['coefficient'])
plt.show()

```


    
![png](images/house-0713/output_49_0.png)
    


## STEP7. 제출


```python
del submission['SalePrice']
submission['SalePrice'] =fpred

submission.to_csv('submission.csv',index=False)
```


```python
submission.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>118146.965100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>144152.058709</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>169665.837534</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>178099.773825</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>191088.830605</td>
    </tr>
  </tbody>
</table>
</div>


