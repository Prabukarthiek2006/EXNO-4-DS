# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```python
import pandas as pd
from scipy import stats
import numpy as np
```
```python
df = pd.read_csv("bmi.csv")
df
```
## Output
<img width="245" height="342" alt="image" src="https://github.com/user-attachments/assets/da24d80d-c10c-4b42-bb0d-aebcb7e6338b" />

```python
df_null_sum=df.isnull().sum()
df_null_sum
```
## Output
<img width="250" height="341" alt="image" src="https://github.com/user-attachments/assets/e7ebf0fa-f1d6-4fd3-9dfc-55575a2a9761" />

```python
df.dropna()
```
## Output
<img width="241" height="344" alt="image" src="https://github.com/user-attachments/assets/22632f92-3c4d-4d73-8092-885661533284" />

```python
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```
## Output
<img width="103" height="48" alt="image" src="https://github.com/user-attachments/assets/b08f6d7c-6a28-4eee-88ad-1e3d47e527b0" />

```python
df1 = df.copy()
```
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df1[['Height', 'Weight']] = sc.fit_transform(df[['Height', 'Weight']])
df1
```
## Output
<img width="280" height="346" alt="image" src="https://github.com/user-attachments/assets/c72cdcbe-eb7b-4f13-ba79-eec39665b73b" />

```python
df2 = df.copy()
```
```python
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
df2[['Height', 'Weight']] = sc.fit_transform(df[['Height', 'Weight']])
df2
```
## Output
<img width="246" height="341" alt="image" src="https://github.com/user-attachments/assets/131b07ae-db27-41c2-abf7-5824cbee0c2a" />

```python
df3 = df.copy()
```
```python
from sklearn.preprocessing import MaxAbsScaler
sc = MaxAbsScaler()
df3[['Height', 'Weight']] = sc.fit_transform(df[['Height', 'Weight']])
df3
```
## Output
<img width="254" height="339" alt="image" src="https://github.com/user-attachments/assets/38cec4b4-1be2-49e7-9ab3-57326cab276d" />

```python
df4 = df.copy()
```
```python
from sklearn.preprocessing import RobustScaler
sc = RobustScaler()
df4[['Height', 'Weight']] = sc.fit_transform(df[['Height', 'Weight']    ])
df4
```
## Output
<img width="261" height="338" alt="image" src="https://github.com/user-attachments/assets/710c2e3b-c558-421b-a922-6b7ce89bdc6d" />

```python
df = pd.read_csv("income(1) (1).csv")
df
```
## Output
<img width="1150" height="344" alt="image" src="https://github.com/user-attachments/assets/3dd09d4c-23ba-4a84-8d06-5d98b76f4bf2" />

```python
df_null_sum=df.isnull().sum()
df_null_sum
```
## Output
<img width="152" height="210" alt="image" src="https://github.com/user-attachments/assets/b5d318cb-06cf-4178-a8ee-0136994bcfea" />

```python
# Chi Square Test
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns]
```

## Output
<img width="687" height="355" alt="image" src="https://github.com/user-attachments/assets/8a907f79-ab03-4f1d-bd52-dc0034b8cd70" />

```python
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
## Output
<img width="605" height="336" alt="image" src="https://github.com/user-attachments/assets/8a2a6522-5dd7-47f9-ab9e-d41a37a67a4f" />

```python
from sklearn.feature_selection import SelectKBest, chi2, f_classif
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_new_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi square test")
print(selected_features_chi2)
```
## Output
<img width="506" height="64" alt="image" src="https://github.com/user-attachments/assets/34c7b421-546d-42c9-9074-fb43266a0c21" />

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss', 'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
## Output
<img width="230" height="75" alt="image" src="https://github.com/user-attachments/assets/42d49d75-e3d1-4e0a-8c36-b5653cdf6bc5" />

```python
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy
```
## Output
<img width="129" height="21" alt="image" src="https://github.com/user-attachments/assets/25b08ca2-fd65-45df-b57c-02b8ade04fb4" />

```python
pip install skfeature-chappers
```
## Output
<img width="797" height="188" alt="image" src="https://github.com/user-attachments/assets/ae0f2090-8c60-4f17-873a-ba39b88d4f23" />

```python
from skfeature.function.similarity_based import fisher_score

categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
## Output
<img width="614" height="343" alt="image" src="https://github.com/user-attachments/assets/d8176b93-f7e6-4048-9780-f28b072354c8" />

```python
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif, k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("Selected features using ANOVA")
print(selected_features_anova)
```
## Output
<img width="590" height="34" alt="image" src="https://github.com/user-attachments/assets/5e6eaa5f-b222-460a-8c7a-0a9297289d54" />


# RESULT:
       # INCLUDE YOUR RESULT HERE
