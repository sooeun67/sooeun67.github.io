---
layout: single
title:  "[딥러닝/AI] TabNet으로 학습한 딥러닝 모델을 SHAP으로 설명력 구현하기 (풀코드구현)"
categories:
  - data science
tags:
  - python
  - Deep Learning
  - XAI
author_profile: false
---

정형 데이터에 최적화된 딥러닝 모델이라고 알려져 있는 [TabNet](https://github.com/dreamquark-ai/tabnet) 으로 학습한 TabNet Regression 모델을 생성하고, 
이 모델을 [SHAP(Shapely Value)](https://shap.readthedocs.io/en/latest/) 을 통해 모델 예측 결과를 최대한 잘 설명해줄 수 방법을 
찾아보고, 결과물을 한 데이터프레임 안에 저장해보자

> TabNet 논문과 SHAP 과 같은 explainable AI 에 관한 내용은 다른 포스트에서 좀더 다룰 예정

# install package
- Colab 환경에서 작성되었음


```python
! pip install pytorch-tabnet ## 설치된 경우에는 실행 안해도 됨
```

```python
! pip install shap ## 설치된 경우에는 실행 안해도 됨 
```


# env setting


```python
### drive mount

from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).

```python
### gpu mapping
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {}".format(DEVICE))
```

    Using cpu



```python
### colab pro + 메모리 사양 확인 
# MemTotal:       13298580 kB
# MemFree:         9764400 kB
# MemAvailable:   12273036 kB

! head -n 3 /proc/meminfo  ### 위의 주석과 다르면.. 실험환경 메모리가 미세하게 다른 것임 ㅠㅜㅠ 
```

    MemTotal:       13297228 kB
    MemFree:         8930012 kB
    MemAvailable:   12369728 kB


# Generate Data & Split Train/Test


```python
### import model 

from sklearn.datasets import load_iris, load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score

from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.metrics import r2_score
import shap
import os
import time

import warnings
warnings.filterwarnings(action='ignore') ### nsample수 20으로 할경우뜨는 warning 메시지 지우기 위해 
```

### Hyperparameter 설정

```python
X_SIZE = 10000 ## 고정 
TEST_SIZE = 20 ### 1000, 2000
COL_SIZE = 15 ### 30, 60
SEED = 2022
N_sample = 'auto'  ## 'auto' or 2048
```

### 데이터 생성
- random 데이터를 뿌려줌


```python
### make data 
np.random.seed(SEED)
x_data = np.random.rand(X_SIZE, COL_SIZE)
df = pd.DataFrame(x_data)
df['target'] = np.random.randint(1000, 50000, size=(X_SIZE, 1))
df
```





  <div id="df-f145a3c1-5e47-47b2-9a6a-75eec5ee37ae">
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.009359</td>
      <td>0.499058</td>
      <td>0.113384</td>
      <td>0.049974</td>
      <td>0.685408</td>
      <td>0.486988</td>
      <td>0.897657</td>
      <td>0.647452</td>
      <td>0.896963</td>
      <td>0.721135</td>
      <td>0.831353</td>
      <td>0.827568</td>
      <td>0.833580</td>
      <td>0.957044</td>
      <td>0.368044</td>
      <td>30906</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.494838</td>
      <td>0.339509</td>
      <td>0.619429</td>
      <td>0.977530</td>
      <td>0.096433</td>
      <td>0.744206</td>
      <td>0.292499</td>
      <td>0.298675</td>
      <td>0.752473</td>
      <td>0.018664</td>
      <td>0.523737</td>
      <td>0.864436</td>
      <td>0.388843</td>
      <td>0.212192</td>
      <td>0.475181</td>
      <td>8187</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.564672</td>
      <td>0.349429</td>
      <td>0.975909</td>
      <td>0.037820</td>
      <td>0.794270</td>
      <td>0.357883</td>
      <td>0.747964</td>
      <td>0.914509</td>
      <td>0.372662</td>
      <td>0.964883</td>
      <td>0.081386</td>
      <td>0.042451</td>
      <td>0.296796</td>
      <td>0.363704</td>
      <td>0.490255</td>
      <td>23806</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.668519</td>
      <td>0.673415</td>
      <td>0.572101</td>
      <td>0.080592</td>
      <td>0.898331</td>
      <td>0.038389</td>
      <td>0.782194</td>
      <td>0.036656</td>
      <td>0.267184</td>
      <td>0.205224</td>
      <td>0.258894</td>
      <td>0.932615</td>
      <td>0.008125</td>
      <td>0.403473</td>
      <td>0.894102</td>
      <td>10204</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.204209</td>
      <td>0.021776</td>
      <td>0.697167</td>
      <td>0.191023</td>
      <td>0.546433</td>
      <td>0.603225</td>
      <td>0.988794</td>
      <td>0.092446</td>
      <td>0.064287</td>
      <td>0.987952</td>
      <td>0.452108</td>
      <td>0.853911</td>
      <td>0.401445</td>
      <td>0.388206</td>
      <td>0.884407</td>
      <td>27620</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>0.017416</td>
      <td>0.824987</td>
      <td>0.246643</td>
      <td>0.141063</td>
      <td>0.184951</td>
      <td>0.384777</td>
      <td>0.722438</td>
      <td>0.279597</td>
      <td>0.194048</td>
      <td>0.816772</td>
      <td>0.070302</td>
      <td>0.708632</td>
      <td>0.497547</td>
      <td>0.113425</td>
      <td>0.302923</td>
      <td>20149</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>0.434106</td>
      <td>0.286644</td>
      <td>0.964673</td>
      <td>0.237779</td>
      <td>0.093510</td>
      <td>0.788614</td>
      <td>0.645321</td>
      <td>0.475191</td>
      <td>0.551407</td>
      <td>0.438434</td>
      <td>0.801701</td>
      <td>0.698005</td>
      <td>0.065917</td>
      <td>0.594159</td>
      <td>0.664846</td>
      <td>9288</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>0.951556</td>
      <td>0.573942</td>
      <td>0.489135</td>
      <td>0.139136</td>
      <td>0.991655</td>
      <td>0.563769</td>
      <td>0.347741</td>
      <td>0.782542</td>
      <td>0.520789</td>
      <td>0.944053</td>
      <td>0.820197</td>
      <td>0.364698</td>
      <td>0.538379</td>
      <td>0.761037</td>
      <td>0.904788</td>
      <td>34624</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>0.800968</td>
      <td>0.732879</td>
      <td>0.651727</td>
      <td>0.610226</td>
      <td>0.644994</td>
      <td>0.756211</td>
      <td>0.247786</td>
      <td>0.620484</td>
      <td>0.464670</td>
      <td>0.879303</td>
      <td>0.108468</td>
      <td>0.580453</td>
      <td>0.742119</td>
      <td>0.414510</td>
      <td>0.988418</td>
      <td>24557</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>0.082800</td>
      <td>0.654022</td>
      <td>0.453132</td>
      <td>0.713547</td>
      <td>0.766718</td>
      <td>0.452666</td>
      <td>0.910464</td>
      <td>0.052970</td>
      <td>0.132754</td>
      <td>0.090441</td>
      <td>0.807935</td>
      <td>0.648001</td>
      <td>0.722958</td>
      <td>0.820611</td>
      <td>0.093902</td>
      <td>9184</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 16 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-f145a3c1-5e47-47b2-9a6a-75eec5ee37ae')"
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
          document.querySelector('#df-f145a3c1-5e47-47b2-9a6a-75eec5ee37ae button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-f145a3c1-5e47-47b2-9a6a-75eec5ee37ae');
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




#### train test split - 위에서 설정한 테스트 개수만큼 split  



```python
X, y = df.drop('target', axis = 1).values, df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, random_state = 2022)
y_train, y_test = y_train.reshape(-1,1), y_test.reshape(-1,1)
```

# Save & Load Model


```python
model_path = f"/content/drive/MyDrive/tabnet_reg_10000/tabnet_reg_{X_SIZE}_{TEST_SIZE}" ### 우선 구글 드라이브에 바로 저장
regressor = TabNetRegressor(verbose=1, seed=2022)

if os.path.isfile(model_path+".zip"):
    print(f"LOAD SAVED MODEL -- {model_path}")
    regressor.load_model(model_path+".zip")
else: 
    print(f"TRANING MODEL & SAVE")
    regressor = TabNetRegressor(verbose=1, seed=2022)
    regressor.fit(X_train=X_train, y_train=y_train, 
                eval_metric=['rmsle'], 
                batch_size = 64, 
                max_epochs=3) ### default epoch # 100 
    regressor.save_model(model_path)

print(regressor)
```

    TRANING MODEL & SAVE
    epoch 0  | loss: 863465970.37419|  0:00:02s
    epoch 1  | loss: 835548792.98065|  0:00:07s
    epoch 2  | loss: 775246533.57419|  0:00:12s
    Successfully saved model at /content/drive/MyDrive/tabnet_reg_10000/tabnet_reg_10000_20.zip
    TabNetRegressor(n_d=8, n_a=8, n_steps=3, gamma=1.3, cat_idxs=[], cat_dims=[], cat_emb_dim=1, n_independent=2, n_shared=2, epsilon=1e-15, momentum=0.02, lambda_sparse=0.001, seed=2022, clip_value=1, verbose=1, optimizer_fn=<class 'torch.optim.adam.Adam'>, optimizer_params={'lr': 0.02}, scheduler_fn=None, scheduler_params={}, mask_type='sparsemax', input_dim=15, output_dim=1, device_name='auto', n_shared_decoder=1, n_indep_decoder=1)



```python
model_path = f"/content/drive/MyDrive/tabnet_reg_10000/tabnet_reg_{X_SIZE}_{TEST_SIZE}" 
regressor = TabNetRegressor(verbose=1, seed=2022)
```


```python
loaded_tabnetregressor = TabNetRegressor()
loaded_tabnetregressor.load_model(model_path+".zip") # 저장한 모델 불러오기
```


```python
loaded_tabnetregressor
```




    TabNetRegressor(n_d=8, n_a=8, n_steps=3, gamma=1.3, cat_idxs=[], cat_dims=[], cat_emb_dim=1, n_independent=2, n_shared=2, epsilon=1e-15, momentum=0.02, lambda_sparse=0.001, seed=2022, clip_value=1, verbose=1, optimizer_fn=<class 'torch.optim.adam.Adam'>, optimizer_params={'lr': 0.02}, scheduler_fn=None, scheduler_params={}, mask_type='sparsemax', input_dim=15, output_dim=1, device_name='auto', n_shared_decoder=1, n_indep_decoder=1)



# Calculate SHAP Value
- `KernelExplainer` 사용 -- Deep Learning model 에는 Permutation 또는 Kernel Explainer가 적합하다


```python
### SHAP value 추출  
print("현재 shap value 확인하는 테스트 데이터 사이즈 -", X_test.shape)
explainer = shap.KernelExplainer(loaded_tabnetregressor.predict, X_test)  # 저장한 모델 넣기

start = time.time() 
shap_values = explainer.shap_values(X_test, nsamples = N_sample)
cell_run_time = time.time() - start
print(cell_run_time)
```

    현재 shap value 확인하는 테스트 데이터 사이즈 - (20, 15)



      0%|          | 0/20 [00:00<?, ?it/s]


    21.929385900497437



```python
### cross check 
base_value = explainer.expected_value[0]
print("base value", base_value)
```

    base value 2976.890240478515


# Merge Predicted Value, Shap Value into Test Dataframe


```python
shap_importance = pd.DataFrame(shap_values[0])
shap_importance.columns = [str(i)+'_SHAPVAL' for i in shap_importance.columns]
shap_importance.loc[:,'TOTAL_SHAPVAL'] = shap_importance.sum(axis=1)
final_report_df = pd.concat([pd.DataFrame(X_test), shap_importance], axis = 1)

y_pred = loaded_tabnetregressor.predict(X_test)
final_report_df['y_pred'] = y_pred
final_report_df['y_pred']  = final_report_df['y_pred'].astype('float64')
final_report_df['base_value']  =  pd.Series([base_value] * X_test.shape[0])
final_report_df['TOTAL_SHAPVAL + base_value'] =  final_report_df['base_value']  + final_report_df['TOTAL_SHAPVAL']
final_report_df

display(final_report_df)
```



  <div id="df-a99a4f05-1980-4c2f-9d0e-6bba01b04059">
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>9_SHAPVAL</th>
      <th>10_SHAPVAL</th>
      <th>11_SHAPVAL</th>
      <th>12_SHAPVAL</th>
      <th>13_SHAPVAL</th>
      <th>14_SHAPVAL</th>
      <th>TOTAL_SHAPVAL</th>
      <th>y_pred</th>
      <th>base_value</th>
      <th>TOTAL_SHAPVAL + base_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.665340</td>
      <td>0.305386</td>
      <td>0.160187</td>
      <td>0.608090</td>
      <td>0.876134</td>
      <td>0.577168</td>
      <td>0.409957</td>
      <td>0.951125</td>
      <td>0.078057</td>
      <td>0.580620</td>
      <td>...</td>
      <td>13.989868</td>
      <td>58.825361</td>
      <td>0.000000</td>
      <td>24.958757</td>
      <td>5.865831</td>
      <td>207.362089</td>
      <td>582.052631</td>
      <td>3558.942871</td>
      <td>2976.89024</td>
      <td>3558.942871</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.896871</td>
      <td>0.016248</td>
      <td>0.043515</td>
      <td>0.174851</td>
      <td>0.286535</td>
      <td>0.426831</td>
      <td>0.938803</td>
      <td>0.245879</td>
      <td>0.652431</td>
      <td>0.358082</td>
      <td>...</td>
      <td>-5.117060</td>
      <td>13.446534</td>
      <td>1.012135</td>
      <td>-7.112571</td>
      <td>0.000000</td>
      <td>-299.820383</td>
      <td>-229.182477</td>
      <td>2747.707520</td>
      <td>2976.89024</td>
      <td>2747.707764</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.195551</td>
      <td>0.534170</td>
      <td>0.054984</td>
      <td>0.067725</td>
      <td>0.899910</td>
      <td>0.864291</td>
      <td>0.928148</td>
      <td>0.663764</td>
      <td>0.529692</td>
      <td>0.069272</td>
      <td>...</td>
      <td>-13.821463</td>
      <td>16.158704</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-797.671184</td>
      <td>-1612.761945</td>
      <td>1364.128174</td>
      <td>2976.89024</td>
      <td>1364.128296</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.780031</td>
      <td>0.958210</td>
      <td>0.060079</td>
      <td>0.127752</td>
      <td>0.121757</td>
      <td>0.998124</td>
      <td>0.619833</td>
      <td>0.771568</td>
      <td>0.357500</td>
      <td>0.250555</td>
      <td>...</td>
      <td>-8.957114</td>
      <td>2.896907</td>
      <td>0.000000</td>
      <td>-2.727257</td>
      <td>1.221694</td>
      <td>-259.071928</td>
      <td>139.967181</td>
      <td>3116.857422</td>
      <td>2976.89024</td>
      <td>3116.857422</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.833555</td>
      <td>0.818795</td>
      <td>0.188312</td>
      <td>0.866473</td>
      <td>0.029219</td>
      <td>0.869027</td>
      <td>0.231443</td>
      <td>0.502989</td>
      <td>0.734818</td>
      <td>0.029553</td>
      <td>...</td>
      <td>-46.340093</td>
      <td>97.512771</td>
      <td>0.000000</td>
      <td>-40.298334</td>
      <td>-10.574096</td>
      <td>14.495060</td>
      <td>128.472308</td>
      <td>3105.362305</td>
      <td>2976.89024</td>
      <td>3105.362549</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.387548</td>
      <td>0.608834</td>
      <td>0.373447</td>
      <td>0.163071</td>
      <td>0.871580</td>
      <td>0.123094</td>
      <td>0.695192</td>
      <td>0.221676</td>
      <td>0.473412</td>
      <td>0.179041</td>
      <td>...</td>
      <td>-5.709209</td>
      <td>-32.263871</td>
      <td>0.000000</td>
      <td>-1.806092</td>
      <td>2.042695</td>
      <td>519.058980</td>
      <td>438.634662</td>
      <td>3415.524658</td>
      <td>2976.89024</td>
      <td>3415.524902</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.051641</td>
      <td>0.885105</td>
      <td>0.505377</td>
      <td>0.459587</td>
      <td>0.717950</td>
      <td>0.021153</td>
      <td>0.329931</td>
      <td>0.348929</td>
      <td>0.030949</td>
      <td>0.122640</td>
      <td>...</td>
      <td>-16.405837</td>
      <td>21.901906</td>
      <td>-2.739079</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-619.011598</td>
      <td>-1625.876813</td>
      <td>1351.013306</td>
      <td>2976.89024</td>
      <td>1351.013428</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.622831</td>
      <td>0.676264</td>
      <td>0.349113</td>
      <td>0.840550</td>
      <td>0.952585</td>
      <td>0.700288</td>
      <td>0.910378</td>
      <td>0.059485</td>
      <td>0.070335</td>
      <td>0.392437</td>
      <td>...</td>
      <td>2.886479</td>
      <td>-40.003732</td>
      <td>0.000000</td>
      <td>7.700811</td>
      <td>0.000000</td>
      <td>174.672702</td>
      <td>53.446185</td>
      <td>3030.336914</td>
      <td>2976.89024</td>
      <td>3030.336426</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.540494</td>
      <td>0.498669</td>
      <td>0.432838</td>
      <td>0.635259</td>
      <td>0.044647</td>
      <td>0.157382</td>
      <td>0.870429</td>
      <td>0.626106</td>
      <td>0.463865</td>
      <td>0.389576</td>
      <td>...</td>
      <td>0.000000</td>
      <td>-22.330007</td>
      <td>0.000000</td>
      <td>-3.186513</td>
      <td>0.000000</td>
      <td>200.520145</td>
      <td>297.537982</td>
      <td>3274.428223</td>
      <td>2976.89024</td>
      <td>3274.428223</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.292736</td>
      <td>0.394259</td>
      <td>0.501476</td>
      <td>0.755071</td>
      <td>0.963806</td>
      <td>0.097967</td>
      <td>0.544224</td>
      <td>0.091328</td>
      <td>0.650178</td>
      <td>0.943432</td>
      <td>...</td>
      <td>17.487106</td>
      <td>-37.783851</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-5.146068</td>
      <td>578.151493</td>
      <td>538.363422</td>
      <td>3515.253174</td>
      <td>2976.89024</td>
      <td>3515.253662</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.807821</td>
      <td>0.546079</td>
      <td>0.826843</td>
      <td>0.606432</td>
      <td>0.196059</td>
      <td>0.102094</td>
      <td>0.280362</td>
      <td>0.134659</td>
      <td>0.952526</td>
      <td>0.791225</td>
      <td>...</td>
      <td>23.460712</td>
      <td>23.505667</td>
      <td>0.000000</td>
      <td>20.740959</td>
      <td>3.940929</td>
      <td>169.248973</td>
      <td>579.097552</td>
      <td>3555.987793</td>
      <td>2976.89024</td>
      <td>3555.987793</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.553643</td>
      <td>0.405755</td>
      <td>0.070046</td>
      <td>0.709272</td>
      <td>0.852507</td>
      <td>0.653668</td>
      <td>0.500025</td>
      <td>0.306107</td>
      <td>0.014521</td>
      <td>0.307509</td>
      <td>...</td>
      <td>-3.137991</td>
      <td>8.399765</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.952945</td>
      <td>222.107463</td>
      <td>587.687885</td>
      <td>3564.578369</td>
      <td>2976.89024</td>
      <td>3564.578125</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.269201</td>
      <td>0.886692</td>
      <td>0.236271</td>
      <td>0.228237</td>
      <td>0.790845</td>
      <td>0.247772</td>
      <td>0.858724</td>
      <td>0.103983</td>
      <td>0.214732</td>
      <td>0.714835</td>
      <td>...</td>
      <td>33.948094</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-4.854956</td>
      <td>-2.832984</td>
      <td>-642.829525</td>
      <td>-1401.765851</td>
      <td>1575.124268</td>
      <td>2976.89024</td>
      <td>1575.124390</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.511182</td>
      <td>0.153969</td>
      <td>0.987458</td>
      <td>0.664299</td>
      <td>0.489893</td>
      <td>0.038254</td>
      <td>0.708350</td>
      <td>0.582648</td>
      <td>0.444579</td>
      <td>0.579134</td>
      <td>...</td>
      <td>8.152289</td>
      <td>3.099521</td>
      <td>0.000000</td>
      <td>4.080772</td>
      <td>6.835309</td>
      <td>167.510164</td>
      <td>599.708881</td>
      <td>3576.599121</td>
      <td>2976.89024</td>
      <td>3576.599121</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.035127</td>
      <td>0.145992</td>
      <td>0.092245</td>
      <td>0.939648</td>
      <td>0.698673</td>
      <td>0.515788</td>
      <td>0.870113</td>
      <td>0.728663</td>
      <td>0.240244</td>
      <td>0.733680</td>
      <td>...</td>
      <td>32.049755</td>
      <td>-13.377773</td>
      <td>0.000000</td>
      <td>-11.680317</td>
      <td>-6.447017</td>
      <td>-558.299584</td>
      <td>-1539.971051</td>
      <td>1436.918945</td>
      <td>2976.89024</td>
      <td>1436.919189</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.458821</td>
      <td>0.365300</td>
      <td>0.913334</td>
      <td>0.213902</td>
      <td>0.705668</td>
      <td>0.444712</td>
      <td>0.655041</td>
      <td>0.711478</td>
      <td>0.109087</td>
      <td>0.047111</td>
      <td>...</td>
      <td>0.000000</td>
      <td>-47.550775</td>
      <td>0.000000</td>
      <td>-8.974647</td>
      <td>-14.749852</td>
      <td>212.535800</td>
      <td>522.897113</td>
      <td>3499.787109</td>
      <td>2976.89024</td>
      <td>3499.787354</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.586684</td>
      <td>0.721589</td>
      <td>0.554982</td>
      <td>0.728450</td>
      <td>0.987413</td>
      <td>0.365390</td>
      <td>0.149811</td>
      <td>0.879073</td>
      <td>0.132804</td>
      <td>0.158433</td>
      <td>...</td>
      <td>-10.001263</td>
      <td>-16.914937</td>
      <td>0.000000</td>
      <td>4.579777</td>
      <td>0.000000</td>
      <td>243.402676</td>
      <td>539.024310</td>
      <td>3515.915283</td>
      <td>2976.89024</td>
      <td>3515.914551</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.447370</td>
      <td>0.856104</td>
      <td>0.490726</td>
      <td>0.512460</td>
      <td>0.901070</td>
      <td>0.723599</td>
      <td>0.946287</td>
      <td>0.362711</td>
      <td>0.341831</td>
      <td>0.226329</td>
      <td>...</td>
      <td>-4.604177</td>
      <td>-30.644028</td>
      <td>0.000000</td>
      <td>20.202807</td>
      <td>10.523801</td>
      <td>161.331920</td>
      <td>237.945941</td>
      <td>3214.836182</td>
      <td>2976.89024</td>
      <td>3214.836182</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.602242</td>
      <td>0.488993</td>
      <td>0.776788</td>
      <td>0.334477</td>
      <td>0.417354</td>
      <td>0.396566</td>
      <td>0.848444</td>
      <td>0.130109</td>
      <td>0.666934</td>
      <td>0.080961</td>
      <td>...</td>
      <td>-5.500653</td>
      <td>5.991046</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.574099</td>
      <td>253.524126</td>
      <td>580.183002</td>
      <td>3557.073242</td>
      <td>2976.89024</td>
      <td>3557.073242</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.459065</td>
      <td>0.059892</td>
      <td>0.831910</td>
      <td>0.959609</td>
      <td>0.553826</td>
      <td>0.954812</td>
      <td>0.142540</td>
      <td>0.463129</td>
      <td>0.410726</td>
      <td>0.211993</td>
      <td>...</td>
      <td>-8.669551</td>
      <td>-23.851995</td>
      <td>3.546563</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>52.351894</td>
      <td>584.539447</td>
      <td>3561.429932</td>
      <td>2976.89024</td>
      <td>3561.429688</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 34 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a99a4f05-1980-4c2f-9d0e-6bba01b04059')"
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
          document.querySelector('#df-a99a4f05-1980-4c2f-9d0e-6bba01b04059 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a99a4f05-1980-4c2f-9d0e-6bba01b04059');
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



# Optional) 시간 소요 그래프 만들기

> Explainer 를 바꿔가면서 시간 소요 테스트를 해보고 싶을 때, log dataframe 을 만들어 추적하면 편하다




```python
cell_run_time
```




    21.929385900497437




```python
log = pd.DataFrame([[TEST_SIZE, COL_SIZE, N_sample, cell_run_time]], columns=['test_size','col_size','nsamples', 'run_time'])
log
# log.to_excel("/content/drive/MyDrive/tabnet_reg_10000/log.xlsx", index=False) # 최초의 log 파일 생성시
# pd.read_excel("/content/drive/MyDrive/tabnet_reg_10000/log.xlsx")
```





  <div id="df-fe4ec609-aa19-4f88-b473-43c804ebf5a0">
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
      <th>test_size</th>
      <th>col_size</th>
      <th>nsamples</th>
      <th>run_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>15</td>
      <td>auto</td>
      <td>21.929386</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-fe4ec609-aa19-4f88-b473-43c804ebf5a0')"
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
          document.querySelector('#df-fe4ec609-aa19-4f88-b473-43c804ebf5a0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-fe4ec609-aa19-4f88-b473-43c804ebf5a0');
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
import os
from openpyxl import load_workbook

cache_path = '/content/drive/MyDrive/tabnet_reg_10000/log.xlsx'

if not os.path.exists(cache_path):  # 파일 없으면 최초 log 파일 만들기
  initial_log = pd.DataFrame([[TEST_SIZE, COL_SIZE, N_sample, cell_run_time]],columns=['test_size','col_size','nsamples', 'run_time'])
  initial_log.to_excel(cache_path, index=False)

else:
  book = load_workbook(cache_path)    # 기존 log 파일 불러오기
  writer = pd.ExcelWriter(cache_path, engine='openpyxl')
  writer.book = book
  writer.sheets = {ws.title: ws for ws in book.worksheets}

  for sheetname in writer.sheets:
      log.to_excel(writer,sheet_name=sheetname, startrow=writer.sheets[sheetname].max_row, index = False,header= False) # append rows to existing excel cache file

  writer.save()
```


```python
pd.read_excel('/content/drive/MyDrive/tabnet_reg_10000/log.xlsx')
```





  <div id="df-15b8b131-a833-41b7-9c38-6064f6da75fd">
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
      <th>test_size</th>
      <th>col_size</th>
      <th>nsamples</th>
      <th>run_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>500</td>
      <td>15</td>
      <td>20</td>
      <td>322.463880</td>
    </tr>
    <tr>
      <th>1</th>
      <td>200</td>
      <td>15</td>
      <td>20</td>
      <td>62.606001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000</td>
      <td>15</td>
      <td>auto</td>
      <td>43426.773710</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100</td>
      <td>15</td>
      <td>auto</td>
      <td>1695.991297</td>
    </tr>
    <tr>
      <th>4</th>
      <td>300</td>
      <td>15</td>
      <td>auto</td>
      <td>18396.789017</td>
    </tr>
    <tr>
      <th>5</th>
      <td>100</td>
      <td>15</td>
      <td>auto</td>
      <td>492.172346</td>
    </tr>
    <tr>
      <th>6</th>
      <td>100</td>
      <td>15</td>
      <td>auto</td>
      <td>498.190434</td>
    </tr>
    <tr>
      <th>7</th>
      <td>200</td>
      <td>15</td>
      <td>auto</td>
      <td>1911.865682</td>
    </tr>
    <tr>
      <th>8</th>
      <td>400</td>
      <td>15</td>
      <td>auto</td>
      <td>7451.173161</td>
    </tr>
    <tr>
      <th>9</th>
      <td>300</td>
      <td>15</td>
      <td>auto</td>
      <td>3855.254270</td>
    </tr>
    <tr>
      <th>10</th>
      <td>500</td>
      <td>15</td>
      <td>auto</td>
      <td>10974.687255</td>
    </tr>
    <tr>
      <th>11</th>
      <td>400</td>
      <td>15</td>
      <td>auto</td>
      <td>6170.332176</td>
    </tr>
    <tr>
      <th>12</th>
      <td>100</td>
      <td>15</td>
      <td>auto</td>
      <td>418.124938</td>
    </tr>
    <tr>
      <th>13</th>
      <td>200</td>
      <td>15</td>
      <td>auto</td>
      <td>1512.140173</td>
    </tr>
    <tr>
      <th>14</th>
      <td>300</td>
      <td>15</td>
      <td>auto</td>
      <td>3359.933324</td>
    </tr>
    <tr>
      <th>15</th>
      <td>400</td>
      <td>15</td>
      <td>auto</td>
      <td>6090.160212</td>
    </tr>
    <tr>
      <th>16</th>
      <td>500</td>
      <td>15</td>
      <td>auto</td>
      <td>9173.967759</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1000</td>
      <td>15</td>
      <td>auto</td>
      <td>36812.828763</td>
    </tr>
    <tr>
      <th>18</th>
      <td>20</td>
      <td>15</td>
      <td>auto</td>
      <td>21.929386</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-15b8b131-a833-41b7-9c38-6064f6da75fd')"
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
          document.querySelector('#df-15b8b131-a833-41b7-9c38-6064f6da75fd button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-15b8b131-a833-41b7-9c38-6064f6da75fd');
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

```
