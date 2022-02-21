---
layout: single
title:  "[Explainable AI] LIME 으로 머신러닝 모델을 해석해보자"
categories: Explainable AI
tags: 
  - python
  - machine learning
  - 데이터분석
toc: false
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }
    
    table.dataframe td {
      text-align: center;
      padding: 8px;
    }
    
    table.dataframe tr:hover {
      background: #b8d1f3; 
    }
    
    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


<a id = 'top-of-notebook'></a>

Explainable AI(설명 가능한 인공지능)의 대표적인 기법인 LIME을 tabular data 에 적용하여 모델을 해석해보자.


## Table of Contents

- [Project Setup](#project-setup)

- [Data Cleansing](#data-cleansing)

- [Data Modelling](#data-modelling)

- [Model Validation](#model-validation)

- [Explainability with LIME](#LIME)


<a id = 'project-setup'></a>

## Project Setup



```python
try:
  import lime
except:
  !pip install lime
  import lime
```


```python
# Importing necessary libraries
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

import lime.lime_tabular

import warnings
warnings.filterwarnings('ignore')
```


```python
# Importing the training data
df_train = pd.read_csv('/content/train.csv')
```


```python
# Viewing the first few rows of the data
df_train.head()
```


  <div id="df-e387599c-3ed1-4016-907c-5b8e87272d9c">
    <div class="colab-df-container">
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-e387599c-3ed1-4016-907c-5b8e87272d9c')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }
    
    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }
    
    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }
    
    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-e387599c-3ed1-4016-907c-5b8e87272d9c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';
    
        async function convertToInteractive(key) {
          const element = document.querySelector('#df-e387599c-3ed1-4016-907c-5b8e87272d9c');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;
    
          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>



[*Return to top of notebook*](#top-of-notebook)


<a id = 'data-cleansing'></a>

## Data Cleansing & Feature Engineering

- Age -> Age Bins: child, teen, young adult, adult, elder, or unknown

- Cabin -> Section: A, B, C, or no cabin



#### 불필요한 칼럼 Drop



```python
df_train.drop(columns = ['PassengerId', 'Name', 'Ticket'], inplace = True)
```

#### "Age" 칼럼 정보에서 "age_bins" feature 만들어 Age칼럼 대체



```python
# Defining / instantiating the necessary variables
age_bins = [-3, -1, 12, 18, 25, 50, 100]
age_labels = ['unknown', 'child', 'teen', 'young_adult', 'adult', 'elder']
# Filling null age values
df_train['Age'] = df_train['Age'].fillna(-2)
# Binning the ages appropriate as defined via the variables above
df_train['Age_Bins'] = pd.cut(df_train['Age'], bins = age_bins, labels = age_labels)
# Dropping the now unneeded 'Age' feature
df_train.drop(columns = 'Age', inplace = True)
```

#### "Cabin" 칼럼 정보에서 맨 앞 한글자씩 따서 "Section" feature 만들어 Cabin 대체

C85 becomes "C"



```python
# Grabbing the first character from the cabin section
df_train['Section'] = df_train['Cabin'].str[:1]
# Filling out the nulls
df_train['Section'].fillna('No Cabin', inplace = True)
# Dropping former 'Cabin' feature
df_train.drop(columns = 'Cabin', inplace = True)
```

#### "Embarked" 칼럼: 결측치 Unknown 으로 채우기



```python
df_train['Embarked'] = df_train['Embarked'].fillna('Unknown')
```

#### Categorical variables 에 대해 One hot encoding



```python
# Defining the categorical features
cat_feats = ['Pclass', 'Sex', 'Age_Bins', 'Embarked', 'Section']
# Instantiating OneHotEncoder
ohe = OneHotEncoder(categories = 'auto')
# Fitting the categorical variables to the encoder
cat_feats_encoded = ohe.fit_transform(df_train[cat_feats])
# Creating a DataFrame with the encoded value information
cat_df = pd.DataFrame(data = cat_feats_encoded.toarray(), columns = ohe.get_feature_names(cat_feats))
```

#### Numerical columns에 대해 Scaling



```python
# Defining the numerical features
num_feats = ['SibSp', 'Parch', 'Fare']
# Instantiating the StandardScaler object
scaler = StandardScaler()
# Fitting the data to the scaler
num_feats_scaled = scaler.fit_transform(df_train[num_feats])
# Creating DataFrame with numerically scaled data
num_df = pd.DataFrame(data = num_feats_scaled, columns = num_feats)
```

#### Encoded & Scaled 된 Dataframes 들을 Concatenate 해서 "X" 라는 training data 만들기



```python
X = pd.concat([cat_df, num_df], axis = 1)
```

#### df_train 에서 target 변수 ("Survived") 를 "y"로 지정



```python
y = df_train[['Survived']]
```

#### Training / Validation Split



```python
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 42)
```

[*Return to top of notebook*](#top-of-notebook)


<a id = 'data-modelling'></a>

## Data Modelling


#### GridSearch 로 optimal 파라미터 찾기



```python
# Defining the random forest parameters
params = {'n_estimators': [10, 50, 100],
          'min_samples_split': [2, 5, 10],
          'min_samples_leaf': [1, 2, 5],
          'max_depth': [10, 20, 50]
         }
         
# Instantiating RandomForestClassifier object and GridSearch object
rfc = RandomForestClassifier()
clf = GridSearchCV(estimator = rfc,
                   param_grid = params)
# Fitting the training data to the GridSearch object
clf.fit(X_train, y_train)
```

<pre>
GridSearchCV(estimator=RandomForestClassifier(),
             param_grid={'max_depth': [10, 20, 50],
                         'min_samples_leaf': [1, 2, 5],
                         'min_samples_split': [2, 5, 10],
                         'n_estimators': [10, 50, 100]})
</pre>

```python
# Displaying the best parameters from the GridSearch object
clf.best_params_
```

<pre>
{'max_depth': 50,
 'min_samples_leaf': 2,
 'min_samples_split': 5,
 'n_estimators': 50}
</pre>
#### Training the model with the ideal params



```python
# Instantiating RandomForestClassifier with ideal params
rfc = RandomForestClassifier(max_depth = 10,
                             min_samples_leaf = 2,
                             min_samples_split = 2,
                             n_estimators = 10)

# Fitting the training data to the model
rfc.fit(X_train, y_train)
```

<pre>
RandomForestClassifier(max_depth=10, min_samples_leaf=2, n_estimators=10)
</pre>
[*Return to top of notebook*](#top-of-notebook)


<a id = 'SHAP'></a>

## LIME 해석 위한 예시 데이터


### Validation set 에서 Survived 와 did not survive 각각 준비



```python
# Contentating X_val and y_val into single df_val set
df_val = pd.concat([X_val, y_val], axis = 1)
```


```python
# Viewing first few rows of df_val to hopefully find 2 good candidates for our study
df_val.head()
```


  <div id="df-f9ecc36f-6335-4bba-b6a3-3876ce0655db">
    <div class="colab-df-container">
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
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
      <th>Sex_female</th>
      <th>Sex_male</th>
      <th>Age_Bins_adult</th>
      <th>Age_Bins_child</th>
      <th>Age_Bins_elder</th>
      <th>Age_Bins_teen</th>
      <th>Age_Bins_unknown</th>
      <th>Age_Bins_young_adult</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>Embarked_Unknown</th>
      <th>Section_A</th>
      <th>Section_B</th>
      <th>Section_C</th>
      <th>Section_D</th>
      <th>Section_E</th>
      <th>Section_F</th>
      <th>Section_G</th>
      <th>Section_No Cabin</th>
      <th>Section_T</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>709</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.432793</td>
      <td>0.767630</td>
      <td>-0.341452</td>
      <td>1</td>
    </tr>
    <tr>
      <th>439</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-0.474545</td>
      <td>-0.473674</td>
      <td>-0.437007</td>
      <td>0</td>
    </tr>
    <tr>
      <th>840</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-0.474545</td>
      <td>-0.473674</td>
      <td>-0.488854</td>
      <td>0</td>
    </tr>
    <tr>
      <th>720</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-0.474545</td>
      <td>0.767630</td>
      <td>0.016023</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.432793</td>
      <td>-0.473674</td>
      <td>-0.422074</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-f9ecc36f-6335-4bba-b6a3-3876ce0655db')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }
    
    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }
    
    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }
    
    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-f9ecc36f-6335-4bba-b6a3-3876ce0655db button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';
    
        async function convertToInteractive(key) {
          const element = document.querySelector('#df-f9ecc36f-6335-4bba-b6a3-3876ce0655db');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;
    
          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




```python
# Noting the two respective people
person_1 = X_val.loc[[720]] # Survived
person_2 = X_val.loc[[439]] # Did not Survive
```

<a id = 'LIME'></a>

# Explainability with LIME

## Instantiate LIME tabular

- Tabular Data 에 대해서 LIME represents a weighted combination of columns
- Args/Parameters
  - 1) training_data: numpy 2d array
  - 2) mode: "classification" or "regression"
  - 3) feature_names: list of names (strings) corresponding to the columns in the training data
  - 4) class_names: list of class names, ordered according to whatever the classifier is using. If not present, class names will be '0', '1', ...
  
```python
# Importing LIME
import lime.lime_tabular

# Defining our LIME explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                        mode = 'classification',
                                                        feature_names = X_train.columns,
                                                        class_names = ['Did Not Survive', 'Survived']
                                                        )
```


```python
# Defining a quick function that can be used to explain the instance passed
predict_rfc_prob = lambda x: rfc.predict_proba(x).astype(float)
```

# 두 가지 케이스에 대한 LIME 해석
### 케이스 1)
 - 왼쪽 패널 ▶ Random Forest Model 은 "Survived" 일 거라고 거의 100% 예측했다
 - 중앙 패널 ▶ 특정 데이터 포인트의 각각의 feature 를 통해 (중요도순) 예측을 설명하고자 한다
   - 이 사람(sample)의 Sex_female feature에 대한 value는 1인데, 0<Sex_female<=1 에 해당되므로 "Survived" 라고 분류될 경향이 더 높다
   - 같은 원리로 이 샘플 Sex_male <= 0 에 해당하기 때문에, "Survived" 라고 분류될 경향이 더 높다
   - 이 샘플은 Age_Bins_child = 1 이므로, Age_Bins_child  > 0.00 에 해당하기 때문에 "Survived" 라고 분류될 경향이 더 높다
   - 그런가하면 Section_E 는 다른 해석을 내놓았다. Section_E= 0 이므로, Section_E <= 0.00 에 해당하기 때문에 "Did Not Survive" 라고 설명한다
- 오른쪽 패널 ▶ 이 샘플의 features & values 를 나타낸다

```python
person_1_lime = lime_explainer.explain_instance(person_1.iloc[0].values,
                                                predict_rfc_prob,
                                                num_features = 10)
person_1_lime.show_in_notebook()
# person_1_lime.save_to_file("person_1.html")
```

![lime_case1](/assets/img/2022-01-12-titanic_lime/lime_case1.png)

## 케이스 2)
 - 왼쪽 패널 ▶ Random Forest Model 은 아래 샘플이 "Did Not Survive" 일 거라고 94% 예측했다
 - 중앙 패널 ▶ 특정 데이터 포인트의 각각의 feature 를 통해 (중요도순) 예측을 설명하고자 한다
   - 이 사람(sample)의 Sex_female feature에 대한 value는 1인데, 0<Sex_female<=1 에 해당되므로 "Did Not Survive" 라고 예측한 것에 대한 설명력이 높다 
   - 같은 원리로 이 샘플 Sex_male <= 0 에 해당하기 때문에, "Did Not Survive" 라고 분류될 경향이 더 높다
   - 이 샘플은 Age_Bins_child = 0 이므로, Age_Bins_child  <= 0.00 에 해당하기 때문에 "Did Not Survive" 라고 분류될 경향이 더 높다
   - 또한, 이 샘플은 Section_E = 0 이므로, Section_E  <= 0.00 에 해당하기 때문에 "Did Not Survive" 라고 분류될 경향이 더 높다
   - 그런가하면 Pclass_3 는 다른 해석을 내놓았다. Pclass_3= 0 이므로, Pclass_3 <= 0.00 에 해당하기 때문에 "Survived" 라고 설명한다
- 오른쪽 패널 ▶ 이 샘플의 features & values 를 나타낸다

```python
person_2_lime = lime_explainer.explain_instance(person_2.iloc[0].values,
                                                predict_rfc_prob,
                                                num_features = 10)
person_2_lime.show_in_notebook()
# person_2_lime.save_to_file("person_2.html")
```

![lime_case2](/assets/img/2022-01-12-titanic_lime/lime_case2.png)

[*Return to top of notebook*](#top-of-notebook)

# To Do & consideration

- Categorical / Numerical 변수들이 많을 때 얼마나 많은 데이터를 갖고 설명해야할지에 대한 선?



# Reference

- [https://towardsdatascience.com/interpreting-black-box-ml-models-using-lime-4fa439be9885](https://towardsdatascience.com/interpreting-black-box-ml-models-using-lime-4fa439be9885)
- [tabular data에 LIME이 어떻게 적용하는지 궁금하다면](https://github.com/marcotcr/lime/blob/master/lime/lime_tabular.py)
