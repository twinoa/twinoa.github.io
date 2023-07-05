---
layout: single
title:  "[ML] 2-3. 모델 선택 - GridSearchCV"
categories: ML
tag: [Machine Learning, Scikit-learn, 모델 선택]
use_math: false
---

## GridSearchCV
데이터 세트를 cross-validation을 위한 학습/테스트 세트로 자동으로 분할한 뒤에 하이퍼 파라미터 그리드에 기술된 모든 파라미터를 순차적으로 적용해 최적의 파라미터를 찾을 수 있게 도와줌

### 주요 파라미터
1. estimator : classifier, regressor, pipeline
2. param_grid : key + 리스트 값을 가지는 딕셔너리가 들어감, estimator 튜닝을 위해 파라미터명과 사용될 여러 파라미터 값을 지정해줌
3. scoring : 예측 성능을 평가할 방법을 지정
4. cv : 교차 검증을 위해 분할되는 학습/테스트 세트의 개수를 지정
5. refit : 가장 최적의 파라미터를 찾은 뒤 입력된 estimator 객체를 해당 하이퍼 파라미터로 재학습 시킴 (기본 : True)


```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# 데이터를 로딩하고 학습 데이터와 테스트 데이터 분리
iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=121)

dtree = DecisionTreeClassifier()

### 파라미터를 딕셔너리 형태로 설정
parameters = {
    'max_depth' : [1, 2, 3], 
    'min_samples_split' : [2, 3]
}
```

### GridSearchCV 결과로 나오는 변수
1. params : 수행할 때마다 적용된 개별 하이퍼 파라미터값
2. rank_test_score : 하이퍼 파라미터별로 성능이 좋은 score 순위
3. mean_test_score : 개별 하이퍼 파라미터별로 CV 폴딩 테스트 세트에 대해 총 수행한 평가 평균값


```python
import pandas as pd
from sklearn.metrics import accuracy_score

# param_grid의 하이퍼 파라미터를 3개의 train, test set fold로 나누어 테스트 수행 설정.
grid_dtree = GridSearchCV(dtree, param_grid=parameters, cv=3, refit=True)

# 붓꽃 학습 데이터로 param_grid의 하이퍼 파라미터를 순차적으로 학습/평가.
grid_dtree.fit(X_train, y_train)

# GridSearchCV 결과를 추출해 DataFrame으로 변환
scores_df = pd.DataFrame(grid_dtree.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score',
          'split0_test_score', 'split1_test_score', 'split2_test_score']]
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
      <th>params</th>
      <th>mean_test_score</th>
      <th>rank_test_score</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'max_depth': 1, 'min_samples_split': 2}</td>
      <td>0.700000</td>
      <td>5</td>
      <td>0.700</td>
      <td>0.7</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'max_depth': 1, 'min_samples_split': 3}</td>
      <td>0.700000</td>
      <td>5</td>
      <td>0.700</td>
      <td>0.7</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{'max_depth': 2, 'min_samples_split': 2}</td>
      <td>0.958333</td>
      <td>3</td>
      <td>0.925</td>
      <td>1.0</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{'max_depth': 2, 'min_samples_split': 3}</td>
      <td>0.958333</td>
      <td>3</td>
      <td>0.925</td>
      <td>1.0</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{'max_depth': 3, 'min_samples_split': 2}</td>
      <td>0.975000</td>
      <td>1</td>
      <td>0.975</td>
      <td>1.0</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>5</th>
      <td>{'max_depth': 3, 'min_samples_split': 3}</td>
      <td>0.975000</td>
      <td>1</td>
      <td>0.975</td>
      <td>1.0</td>
      <td>0.95</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('GridSearchCV 최적 파라미터 : ', grid_dtree.best_params_)
print('GridSearchCV 최고 정확도 : {0:.4f}'.format(grid_dtree.best_score_))
```

    GridSearchCV 최적 파라미터 :  {'max_depth': 3, 'min_samples_split': 2}
    GridSearchCV 최고 정확도 : 0.9750
    


```python
# GridSearchCV의 refit으로 이미 학습된 estimator 반환
estimator = grid_dtree.best_estimator_

# GridSearchCV의 best_estimator_는 이미 최적 학습이 됐으므로 별도 학습이 필요 없음
pred = estimator.predict(X_test)
print('테스트 데이터 세트 정확도 : {0:.4f}'.format(accuracy_score(y_test, pred)))
```

    테스트 데이터 세트 정확도 : 0.9667
    
