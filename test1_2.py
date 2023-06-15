#제1유형
#qsec 컬럼을 최소최대척도(Min-Max Scale)로 변환한 후 0.5보다 큰 값을 가지는 레코드 수를 구하시오

# 출력을 원할 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가

# 데이터 파일 읽기 예제
import pandas as pd
a = pd.read_csv('data/mtcars.csv', index_col=0)

# 사용자 코딩
from sklearn.preprocessing import MinMaxScaler
scaler = ['qsec']
min = MinMaxScaler()
min.fit(a[scaler])
a[scaler] = min.transform(a[scaler])

# print(a[scaler])

result = a[a['qsec']>0.5]

# 답안 제출 예시
# print(평균변수값)
print('결과값', len(result))


########################################
#작업형 2

# # 데이터 파일 읽기 예제
# import pandas as pd
# X_test = pd.read_csv("data/X_test.csv")
# X_train = pd.read_csv("data/X_train.csv")
# y_train = pd.read_csv("data/y_train.csv")

# 사용자 코딩
#1. 결측치 제거
X_train['환불금액'] = X_train['환불금액'].fillna(0)
X_test['환불금액'] = X_test['환불금액'].fillna(0)

#2. 라벨 인코더 (수정부분)
label = ['주구매상품', '주구매지점']
from sklearn.preprocessing import LabelEncoder 
X_train[label] = X_train[label].apply(LabelEncoder().fit_transform)
X_test[label] = X_test[label].apply(LabelEncoder().fit_transform)


#print(X_train['주구매지점'].value_counts())

#3. 카테고리 변환, 더미
category = ['주구매지점']
for i in category:
    X_train[i] = X_train[i].astype('category')
    X_test[i] = X_test[i].astype('category')
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

#4. 파생변수 생성
X_train['총구매액_qcut'] = pd.qcut(X_train['총구매액'], 5, labels = False)
X_test['총구매액_qcut'] = pd.qcut(X_test['총구매액'], 5, labels = False)

#5. 스케일 작업
from sklearn.preprocessing import MinMaxScaler
scaler = ['총구매액', '최대구매액', '환불금액', '내점일수', '내점당구매건수', '주말방문비율', '구매주기']
min = MinMaxScaler()
min.fit(X_train[scaler])
X_train[scaler] = min.transform(X_train[scaler])
X_test[scaler] = min.transform(X_test[scaler])

#6. 데이터 분리
from sklearn.model_selection import train_test_split 
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train['gender'], test_size=0.2, random_state=42, stratify=y_train['gender'])



#7. 모형학습 = Logistic Regression / Random Forest / Voting 8. 앙상블
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(X_train, y_train)
pred1 = pd.DataFrame(model1.predict_proba(X_valid))

from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier()
model2.fit(X_train, y_train)
pred2 = pd.DataFrame(model2.predict_proba(X_valid))

from sklearn.ensemble import VotingClassifier
model3 = VotingClassifier(estimators=[('logistic', model1), ('random', model2)], voting='soft')
model3.fit(X_train, y_train)
pred3 = pd.DataFrame(model3.predict_proba(X_valid))

#9. 모형평가
from sklearn.metrics import roc_auc_score
print('로지스틱', roc_auc_score(y_valid, pred1.iloc[:,1]))
print('랜덤포레스트', roc_auc_score(y_valid, pred2.iloc[:,1]))
print('보팅', roc_auc_score(y_valid, pred3.iloc[:,1]))

#10. 하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators':[50,100], 'max_depth':[4,6]}
model4 = RandomForestClassifier()
clf = GridSearchCV(estimator= model4, param_grid = parameters, cv=3)
clf.fit(X_train, y_train)
print('최적의 파라메터', clf.best_params_)

# clf.fit(X_train, y_train):[50, 100], 'max_depth':[4,6]}
# model4 = RandomForestClassifier()
# clf = GridSearchCV(estimator= model4, param_grid=parameters, cv=3)
# print('최적의 파라메터', clf.best_params_) 

#11. 파일저장
result = pd.DataFrame(model3.predict_proba(X_test))
result = result.iloc[:,1]
pd.DataFrame({'custid': X_test['cust_id'], 'gender': result}).to_csv('003000000.csv', index=False)

#확인
check = pd.read_csv('003000000.csv')
print(check.head())
