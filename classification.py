#데이터 파일 읽어오기

import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

df = sns.load_dataset('titanic')
#훈련, 테스트용 데이터 분리 - random_state는 주어지는 난수값
X_train, X_test, y_train, y_test = train_test_split(df, df['survived'], test_size=0.2, random_state=42,
stratify=df['survived'])
X_train = X_train.drop(['alive','survived'], axis=1) #axis = 0이면 행 제거 1이면 열 제거 
X_test = X_test.drop(['alive','survived'], axis=1)

#####데이터 전처리#########
                
#1. 결측치 채우기
print(X_train.head())
print(X_train.isna().sum()) 
# 결측치 age - 숫자형 => 평균, deck - 문자형=> 가장 많은 분포

#분포도 알아보기
print('deck', X_train['deck'].value_counts())
print('embarked', X_train['embarked'].value_counts())
print('embark_town', X_train['embark_town'].value_counts())

#age - 평균값으로 대체 
missing = ['age']
for i in missing:
    X_train[i] = X_train[i].fillna(X_train[i].mean())
    X_test[i] = X_test[i].fillna(X_test[i].mean())
    
#deck - 가장 많은 값으로 대체 
X_train['deck'] = X_train['deck'].fillna('C')
X_test['deck'] = X_test['deck'].fillna('C')

X_train['embarked'] = X_train['embarked'].fillna('S')
X_test['embarked'] = X_test['embarked'].fillna('S')

X_train['embark_town'] = X_train['embark_town'].fillna('Southampton')
X_test['embark_town'] = X_test['embark_town'].fillna('Southampton')

print(X_train.head())
print(X_train.isna().sum())



#2. 라벨 인코딩 - 문자열을 숫자로 변환
from sklearn.preprocessing import LabelEncoder
label = ['sex', 'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alone']
X_train[label] = X_train[label].apply(LabelEncoder().fit_transform)
X_test[label] = X_test[label].apply(LabelEncoder().fit_transform)

print(X_train.head())

#3. 데이터 타입변환, 더미 처리
print(X_train.dtypes)
      
dtype=['pclass', 'sex', 'class']
for i in X_train[dtype]:
    X_train[i]= X_train[i].astype('category')
for i in X_test[dtype]:
    X_test[i] = X_test[i].astype('category')
      
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
print(X_train.head())

#4. 파생변수 <- 연속형변수
#일단 예를 들어 age 하나만 작업해봄. 5개의 구간으로 나눔 - 모델의 성능 향상 

X_train['age_qcut'] = pd.qcut(X_train['age'], 5, labels=False)
X_test['age_qcut'] = pd.qcut(X_test['age'], 5, labels=False)
print(X_train.head())

#5. 스케일 
#age, fare의 단위를 맞춰준다. 0~1사이로.

from sklearn.preprocessing import MinMaxScaler
scaler = ['age', 'fare']
min = MinMaxScaler()
min.fit(X_train[scaler])

X_train[scaler] = min.transform(X_train[scaler])
X_test[scaler] = min.transform(X_test[scaler])

print(X_train.head())

#6. 데이터 분리 작업
#훈련용과 검증용 데이터를 만드는 작업
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42, 
stratify = y_train )
#분리 작업 확인 - train : 569개, valid: 143개로 데이터 분리 
print('X_train', X_train.shape)
print('X_valid', X_valid.shape)

#############모형학습 및 평가#####################
#7. 모형학습, 앙상블 
#로지스틱 회귀
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(X_train, y_train)
      
pred1 = pd.DataFrame(model1.predict_proba(X_valid)) 

#랜덤포레스트 
from sklearn.ensemble import RandomForestClassifier 
model2 = RandomForestClassifier()
model2.fit(X_train, y_train)
      
pred2 = pd.DataFrame(model2.predict_proba(X_valid))
                                           
                                    
# 두가지 모델을 voting으로 합치기 -> 확률로 계산하기 때문에 soft option
#결과값은 0~1까지의 확률값으로 나온다. 제출할 때는 1번에 대한 열만 제출해야함.
from sklearn.ensemble import VotingClassifier 
model3 = VotingClassifier(estimators=[('logistic', model1),('random', model2)], voting='soft') 
model3.fit(X_train, y_train)
pred3= pd.DataFrame(model3.predict_proba(X_valid))  

print(pred3)

#9. 모형 평가
from sklearn.metrics import roc_auc_score
print('로지스틱', roc_auc_score(y_valid, pred1.iloc[:,1]))
print('랜덤 포레스트', roc_auc_score(y_valid, pred2.iloc[:,1]))
print('Voting', roc_auc_score(y_valid, pred3.iloc[:,1]))

# 로지스틱 0.859400826446281
# 랜덤 포레스트 0.86301652892562
# Voting 0.8722107438016528 - 성능이 올라감

#10. 하이퍼 파라미터 튜닝
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators': [50, 100], 'max_depth': [4,6]}
model5 = RandomForestClassifier()
clf = GridSearchCV(estimator = model5, param_grid = parameters, cv=3)
clf.fit(X_train, y_train)
print('최적의 파라미터', clf.best_params_)

#11. 파일 저장
result = pd.DataFrame(model3.predict_proba(X_test))
result = result.iloc[:,1]
pd.DataFrame({'id': X_test.index, 'result':result}).to_csv('00300.csv', index=False)

#확인
check = pd.read_csv('00300.csv')
print(check.head())