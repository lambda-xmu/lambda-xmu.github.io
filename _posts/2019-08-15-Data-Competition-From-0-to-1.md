---
layout:     post
title:      2019-08-15-Data-competition-From-0-to-1-Part-I
subtitle:   "Credit Fraud Detector Example"
date:       2018-08-15 12:00:00
author:     "lambda"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
- Data Competition
- EDA
- Unbalance
- Metrics
- Sampling
---

> Concat: [github: lambda_xmu](https://github.com/lambda-xmu)

## 1. Data competition Introduction
A typical data science process might look like this:

- Project Scoping / Data Collection
- Exploratory Analysis
- Data Cleaning
- Feature Engineering
- Model Training (including cross-validation to tune hyper-parameters)
- Project Delivery / Insights
@import "picture/data competition.png"

## 2. Example: Credit Fraud Detector
**Competition Describe**
* The datasets contains transactions made by credit cards in September 2013 by european cardholders.
* Features V1, V2, ... V28 are the principal components obtained with PCA
* Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset.
* The feature ‘Amount’ is the transaction Amount.
* Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

### EDA(Exploratory Data Analysis)
- Gathering a basic sense of data
- Example:
  - Shape(Traing Size, Test size)
  - Label(Binary or Multi or Regression, Distribution)
  - Columns(Meaning, Numerical or Time or Category)
  - `Null` Values, how to deal with
  - Numerical variable: Distribution
  - Outliers

```python
# import library
import pandas as pd
pd.set_option('display.max_column',100)
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 14, 10
# plt.figure(figsize=(14, 10))
import numpy as np
import seaborn as sns

%matplotlib inline
# 可以省略 plt.show()

import warnings
warnings.filterwarnings('ignore')
```

```python
# simple ead ways
data = pd.read_csv('data/creditcard.csv')
data.head()
data.shape
data.info()
data.columns
data.describe()
data['Class'].value_counts()
data.isnull().sum()
data.isnull().sum().max() # 包含缺失最多的一列有多少
data.isnull().any()
data.isnull().values.any() # 数据中是否有缺失值
```

```python
# Amount-histogram
Fraud_transacation = data[data["Class"]==1]
Normal_transacation= data[data["Class"]==0]
plt.figure(figsize=(10,6))
plt.subplot(121)
Fraud_transacation.Amount.plot.hist(title="Fraud Transacation")
plt.yscale('log')
plt.subplot(122)
Normal_transacation.Amount.plot.hist(title="Normal Transaction")
plt.yscale('log')

# Amount-Time
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(Fraud.Time, Fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(Normal.Time, Normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()

# correlation
correlation_matrix = data.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(correlation_matrix,vmax=0.8,square = True)
plt.show()

## After sampling ?
```

```python
data['hour'] = data['Time'].apply(lambda x: np.ceil(float(x)/3600) % 24)
data.pivot_table(values='Amount',index='hour',columns='Class',aggfunc='count')
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot_table.html

bins = np.arange(data['hour'].min(),data['hour'].max()+2)
# plt.figure(figsize=(15,4))
sns.distplot(data[data['Class']==0.0]['hour'],
             norm_hist=True,
             bins=bins,
             kde=False,
             color='b',
             hist_kws={'alpha':.5},
             label='Legit')
sns.distplot(data[data['Class']==1.0]['hour'],
             norm_hist=True,
             bins=bins,
             kde=False,
             color='r',
             label='Fraud',
             hist_kws={'alpha':.5})
plt.xticks(range(0,24))
plt.legend()
```


```python
# https://github.com/pandas-profiling/pandas-profiling
import pandas_profiling

profile = df.profile_report(title="Credit Fraud Detector")
profile.to_file(output_file=Path("./credit_fraud_detector.html"))
```
**EDA Conclusion**
>- The data set is highly skewed, consisting of 492 frauds in a total of 284,807 observations. This resulted in only 0.172% fraud cases.
>- There is no missing value in the dataset.
>- The ‘Time’ and ‘Amount’ features are not transformed data.

#### Why taking log transformation of continuous variables?
Amount distribution is "power law", meaning that the vast majority of amount are small and very few are big.

$$\begin{array}{c}{\log \left(10^{4}\right)=4 * \log (10)} \\ {\log \left(10^{3}\right)=3 * \log (10)} \\ {10^{4}-10^{3}} \\ {4-3}\end{array}$$

which transforms a huge difference in a smaller one. **Logarithm naturally reduces the dynamic range of a variable so the differences are preserved while the scale is not that dramatically skewed.**




### Outliers Detection
- **Point anomalies**: A single instance of data is anomalous if it's too far off from the rest.
- **Contextual anomalies**: The abnormality is context specific. This type of anomaly is common in time-series data.

<img src="img/anomaly.png" width = "300" height = "300" alt="图片名称" align=center />

$$\mathrm{IQR}=Q_{3}-Q_{1}$$

$$\text{Outliers}: >Q_3+k \cdot IQR$$

$$\text{Outliers}: <Q_1-k \cdot IQR$$

The higher $k$ is (ex: 3), the less outliers will detect, and the lower $k$ is (ex: 1.5) the more outliers it will detect.

<p align="left" style="font-size: 25px;">
<font color=red>
We want to focus more on "extreme outliers" rather than just outliers.
</font>
</p>

### Unbalance
```python
count_classes = pd.value_counts(df['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
```

<img src="img/class_distribution.png" width = "400" height = "300" alt="图片名称" align=center />

1. Collect more data
2. Using the weights parameters
```python
LogisticRegression(class_weight='balanced')
# How to choose weights?
```
3. Changing the performance metric:
  2.1 `Precision`, `Recall`
  2.2 `F1-score`
  2.3 `ROC` curves
4. Resampling the dataset
  3.1 `OVER-sampling`
  3.2 `UNDER-sampling`
  3.3 `SMOTE`

### Metrics
**Confusion Matrix**
<img src="img/prediction.png" width = "400" height = "300" alt="图片名称" align=center />
- `True Positives` : The cases in which we predicted YES and the actual output was also YES.
- `True Negatives` : The cases in which we predicted NO and the actual output was NO.
- `False Positives` : The cases in which we predicted YES and the actual output was NO.
- `False Negatives` : The cases in which we predicted NO and the actual output was YES.
<img src="img/Confusion Matrix2.png" width = "600" height = "200" alt="图片名称" align=center />

```python
## homework ##
def find_TP(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))
def find_FN(y_true, y_pred):
    # counts the number of false negatives (y_true = 1, y_pred = 0)
    return # your code here
def find_FP(y_true, y_pred):
    # counts the number of false positives (y_true = 0, y_pred = 1)
    return # your code here
def find_TN(y_true, y_pred):
    # counts the number of true negatives (y_true = 0, y_pred = 0)
    return # your code here

import numpy as np
def find_conf_matrix_values(y_true,y_pred):
    # calculate TP, FN, FP, TN
    TP = find_TP(y_true,y_pred)
    FN = find_FN(y_true,y_pred)
    FP = find_FP(y_true,y_pred)
    TN = find_TN(y_true,y_pred)
    return TP,FN,FP,TN
def my_confusion_matrix(y_true, y_pred):
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
    return np.array([[TN,FP],[FN,TP]])
```

```python
from sklearn.metrics import confusion_matrix
confusion_matrix(label, prediction)
```

**Accuracy**
$$\text{Accuracy}=\frac{\text{True Positives}+\text{False Negatives}}{\text{Total Number of Predictions}}$$
- It works well only if there are equal number of samples belonging to each class.

```python
# homework
def my_accuracy_score(y_true, y_pred):
    # calculates the fraction of samples predicted correctly
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
    return # your code here
```

```python
from sklearn.metrics import accuracy_score
accuracy_score(label, prediction)
```
**Precision**
$$\text{Precision}=\frac{\text{True Positives}}{\text{True Positives}+\text{False Positives}}$$

```python
# homework
def my_precision_score(y_true, y_pred):
    # calculates the fraction of predicted positives samples that are actually positive
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
    return # your code here
```

```python
from sklearn.metrics import precision_score
precision_score(label, prediction)
```

**Recall**
$$\text{Recall}=\frac{\text{True Positives}}{\text{True Positives}+\text{False Negatives}}$$
```python
# homework
def my_recall_score(y_true, y_pred):
    # calculates the fraction of positive samples predicted correctly
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
    return # your code here
```

```python
from sklearn.metrics import recall_score
recall_score(label, prediction)
```

**F1 Score**
The f1 score is the harmonic mean(调和平均) of recall and precision, with a higher score as a better model.
$$F 1=\frac{2}{\frac{1}{\text { precision }}+\frac{1}{\text { recall }}}=\frac{2 * \text { (precision * recall) }}{\text { precision }+\text {recall}}$$
```python
# homework
def my_f1_score(y_true, y_pred):
    # calculates the F1 score
    recall = my_recall_score(y_true,y_pred)
    precision = my_precision_score(y_true,y_pred)
    return # your code here
```

```python
from sklearn.metrics import f1_score
f1_score(label, prediction)
```
<font color=red>We often assume that we defined a threshold of 0.5 for selecting which samples are predicted as positive. If we change this threshold the performance metrics will change</font> It would be nice to be able to evaluate the performance of a model without the need to select an arbitrary threshold.  This is precisely what AUC-ROC is providing.

**Area Under Curve (AUC)**--Area Under the curve of the Receiver Operating Characteristic (AUROC)

AUC is used for binary classification problem.

- True Positive Rate (Sensitivity) : the proportion of positive data points that are correctly considered as positive, with respect to all positive data points.$$\text{True Positive Rate}=\frac{\text{True Positives}}{\text{True Positives}+\text{False Negatives}}$$
- False Positive Rate (Specificity) : the proportion of negative data points that are mistakenly considered as positive, with respect to all negative data points.$$\text{False Positive Rate}=\frac{\text{False Positives}}{\text{False Positives}+\text{True Negatives}}$$

`FPR` and `TPR` bot hare computed at threshold values such as $(0.00, 0.02, 0.04, …., 1.00)$ and a graph is drawn. `AUC` is the area under the curve of plot `False Positive Rate` vs `True Positive Rate` at different points in $[0, 1]$. The resulting curve is called ROC curve (Receiver Operating Characteristic curve).

```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(label, prediction)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr)
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(label, prediction)
```

<img src="img/auc.png" width = "500" height = "400" alt="图片名称" align=center />

```python
from numba import jit

@jit
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc
```

```python
## test speed
y_true = np.random.randint(0,2,1000000)
y_pred = np.random.rand(1000000)

fast_auc(y_true, y_pred)

from sklearn.metrics import roc_auc_score
roc_auc_score(y_true, y_pred)

%timeit fast_auc(y_true, y_pred)
%timeit roc_auc_score(y_true, y_pred)
```

You can learn more about [ROC on the Wikipedia page](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) and [What does AUC stand for and what is it?](https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it).

**Mean Absolute Error(MAE)**
$$\text {MAE}=\frac{1}{N} \sum_{j=1}^{N}\left|y_{j}-\hat{y}_{j}\right|$$

**Mean Squared Error(MSE)**
$$\text {MSE}=\frac{1}{N} \sum_{j=1}^{N}\left(y_{j}-\hat{y}_{j}\right)^{2}$$

**Log Loss**
`AUC` only takes into account **the order of probabilities** and hence it does not take into account the model’s capability to predict higher probability for samples more likely to be positive.


$$\text {Log Loss}=-\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{M} y_{i j} * \log \left(p(y_{i j})\right)$$
- $y_{ij}$: whether sample $i$ belongs to class $j$ or not
- $p(y_{i j})$: the probability of sample $i$ belonging to class $j$


$$\text {Log Loss}=-\frac{1}{N} \sum_{i=1}^{N} y_{i} \cdot \log \left(p\left(y_{i}\right)\right)+\left(1-y_{i}\right) \cdot \log \left(1-p\left(y_{i}\right)\right)$$

### Resampling
**Under-Sampling**: samples from the majority class
**Over-Sampling**: adding more examples from the minority class

**Under-Sampling Drawback**: Removing information that may be valuable. This could lead to underfitting and poor generalization to the test set.
<img src="img/resampling.png" width = "700" height = "200" alt="图片名称" align=center />
```python
### Create A Small Unbalanced Sample Dataset
from sklearn.utils import resample
from sklearn.datasets import make_classification

X, y = make_classification(
    n_classes=2, class_sep=1.5, weights=[0.9, 0.1],
    n_informative=3, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1,
    n_samples=1000, random_state=10
)

df = pd.DataFrame(X)
df['target'] = y
df.target.value_counts().plot(kind='bar', title='Count (target)')

### Oversample minority class
minority_data = df[df.target==1]
magority_data = df[df.target==0]

# oversample minority
upsampled_data = resample(minority_data,
                          replace=True, # sample with replacement
                          n_samples=len(magority_data), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([upsampled_data, magority_data])

# check new class counts
upsampled.target.value_counts()

### Undersample majority class
downsampled_data = resample(magority_data,
                            replace = False, # sample without replacement
                            n_samples = len(minority_data), # match minority n
                            random_state = 27) # reproducible results

# combine minority and downsampled majority
downsampled = pd.concat([minority_data, downsampled_data])

# checking counts
downsampled.target.value_counts()
```

**Imbalanced-Learn**
```python
def plot_2d_space(X_train, y_train,X=X,y=y ,label='Classes'):
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']

    fig,(ax1,ax2)=plt.subplots(1,2, figsize=(10,5))

    for l, c, m in zip(np.unique(y), colors, markers):
        ax1.scatter(
            X_train[y_train==l, 0],
            X_train[y_train==l, 1],
            c=c, label=l, marker=m
        )
    for l, c, m in zip(np.unique(y), colors, markers):
        ax2.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )

    ax1.set_title(label)
    ax2.set_title('original data')
    plt.legend(loc='upper right')
    plt.show()

import imblearn
### Random under-sampling with imbalanced-learn
from imblearn.under_sampling import RandomUnderSampler

ran = RandomUnderSampler(return_indices=True) ##intialize to return indices of dropped rows
X_rs,y_rs,dropped = ran.fit_sample(X,y)

print("The number of removed indices are ",len(dropped))
plot_2d_space(X_rs, y_rs, X, y,'Random under sampling')

### Random over-sampling with imbalanced-learn
from imblearn.over_sampling import RandomOverSampler

ran=RandomOverSampler()
X_ran,y_ran= ran.fit_resample(X,y)

print('The new data contains {} rows '.format(X_ran.shape[0]))
plot_2d_space(X_ran,y_ran,X,y,'over-sampled')
```
**Under-sampling: Tomek links**
**Tomek links** are pairs of very close instances, but of opposite classes. Removing the instances of the majority class of each pair increases the space between the two classes, facilitating the classification process.

<img src="img/tomek.png" width = "700" height = "200" alt="图片名称" align=center />

```python
from imblearn.under_sampling import TomekLinks

tl = TomekLinks(return_indices=True, ratio='majority')
X_tl, y_tl, id_tl = tl.fit_sample(X, y)

#print('Removed indexes:', id_tl)
plot_2d_space(X_tl, y_tl,X,y, 'Tomek links under-sampling')
```
**Over-sampling: SMOTE**
**SMOTE (Synthetic Minority Oversampling TEchnique)** consists of synthesizing elements for the minority class, based on those that already exist. It works randomly picingk a point from the minority class and computing the k-nearest neighbors for this point. The synthetic points are added between the chosen point and its neighbors.

<img src="img/smote 2.png" width = "700" height = "200" alt="图片名称" align=center />

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(ratio='minority')
X_sm, y_sm = smote.fit_sample(X, y)

plot_2d_space(X_sm, y_sm,X,y, 'SMOTE over-sampling')
```

### Cross-validation: evaluating estimator performance
Trained model would have a perfect score on training data but would fail to predict anything useful on yet-unseen data. This situation is called `overfitting`. To avoid it, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set `X_test`, `y_test`. Here is a flowchart of typical cross validation workflow in model training.

<img src="picture/grid_search_workflow.png" width = "700" height = "400" alt="图片名称" align=center />

**`k-folds`**:
- A model is trained using `k-1` of the folds as training data;
- the resulting model is validated on the remaining part of the data.

The performance measure reported by `k-fold` cross-validation is then the average of the values computed in the loop. This approach can be computationally expensive, but does not waste too much data.

<img src="picture/grid_search_cross_validation.png" width = "700" height = "500" alt="图片名称" align=center />

#### Cross validation iterators
**Cross-validation iterators for i.i.d. data**
**`K-fold`**
<img src="picture/sphx_glr_plot_cv_indices_0041.png" width = "700" height = "350" alt="图片名称" align=center />

```python
from sklearn.model_selection import KFold

X = ["a", "b", "c", "d"]
kf = KFold(n_splits=2)
for train, test in kf.split(X):
    print("%s %s" % (train, test))
```

**`Random permutations cross-validation a.k.a. Shuffle & Split`**
<img src="picture/sphx_glr_plot_cv_indices_0061.png" width = "700" height = "350" alt="图片名称" align=center />
```python
from sklearn.model_selection import ShuffleSplit
X = np.arange(10)
ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
for train_index, test_index in ss.split(X):
    print("%s %s" % (train_index, test_index))
```

**Cross-validation iterators with stratification based on class labels**
Some classification problems can exhibit a large imbalance in the distribution of the target classes.

**`Stratified k-fold`**
<img src="picture/sphx_glr_plot_cv_indices_0071.png" width = "700" height = "350" alt="图片名称" align=center />
```python
from sklearn.model_selection import StratifiedKFold

X = np.ones(10)
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
    print("%s %s" % (train, test))
```

**`Stratified Shuffle Split`**
<img src="picture/sphx_glr_plot_cv_indices_0091.png" width = "700" height = "350" alt="图片名称" align=center />

#### Cross-validation iterators for grouped data
The i.i.d. assumption is broken if the underlying generative process yield groups of dependent samples.

**`Group k-fold`**
GroupKFold is a variation of k-fold which ensures that the same group is not represented in both testing and training sets.
<img src="picture/sphx_glr_plot_cv_indices_0051.png" width = "700" height = "350" alt="图片名称" align=center />
```python
from sklearn.model_selection import GroupKFold

X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]

gkf = GroupKFold(n_splits=3)
for train, test in gkf.split(X, y, groups=groups):
    print("%s %s" % (train, test))
```

**`Group Shuffle Split`**
<img src="picture/sphx_glr_plot_cv_indices_0081.png" width = "700" height = "350" alt="图片名称" align=center />
```python
from sklearn.model_selection import GroupShuffleSplit

X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 0.001]
y = ["a", "b", "b", "b", "c", "c", "c", "a"]
groups = [1, 1, 2, 2, 3, 3, 4, 4]
gss = GroupShuffleSplit(n_splits=4, test_size=0.5, random_state=0)
for train, test in gss.split(X, y, groups=groups):
    print("%s %s" % (train, test))
```

**`Leave P Groups Out`**
```python
from sklearn.model_selection import LeavePGroupsOut

X = np.arange(6)
y = [1, 1, 1, 2, 2, 2]
groups = [1, 1, 2, 2, 3, 3]
lpgo = LeavePGroupsOut(n_groups=2)
for train, test in lpgo.split(X, y, groups=groups):
    print("%s %s" % (train, test))
```

#### Cross validation of time series data
`k-fold` 假设数据是 `iid` 的，但是时间序列存在相关性，因此不能简单使用 `k-fold`。
<img src="picture/sphx_glr_plot_cv_indices_0101.png" width = "700" height = "350" alt="图片名称" align=center />
```python
from sklearn.model_selection import TimeSeriesSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])
tscv = TimeSeriesSplit(n_splits=3)
for train, test in tscv.split(X):
    print("%s %s" % (train, test))
```

### Modeling
```python
from sklearn.preprocessing import StandardScaler
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
data = data.drop(['Amount'],axis=1)
```
```python
# train_test_split
from sklearn.cross_validation import train_test_split
# Whole dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C = c_param, penalty = 'l1')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
# compare y_test with y_pred
```
