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


print(X_train.isna().sum())



#2. 라벨 인코딩
from sklearn.preprocessing import LabelEncoder
