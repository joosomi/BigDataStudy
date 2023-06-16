#################T검정 - 독립 소표본, 단순 평균##################
# 표본 평균 계산 - pandas
# 검정 통계량 계산  - scipy.stats.ttest_1samp (T-test for One group)

#1. 표본평균 (단순 표본 평균) // 치료 후 혈압
simple_mean = a['bp_after'].mean().round(2)
print(simple_mean)
# print(type(simple_mean))


#2.검정통계량 p-value
#치료 후의 혈압이 160보다 작다를 가설 검정하는 경우
import scipy
from scipy import stats

alpha = 0.05
tstat = scipy.stats.ttest_1samp(a['bp_after'], popmean = 160, alternative ='less')
print(tstat)
print('t-value:', tstat.statistic.round(4))
print('p-valeu', tstat.pvalue.round(4))
# t-value: -6.6771
# p-value: 0.0


#3. 가설 검정 치료 후 혈압의 평균 = 160 (대립가설

#3.1 critical value를 이용하는 방법 #scipy.stats.t 의 ppf
# 자유도 n-1
cv_simple_mean = scipy.stats.t.ppf(q = alpha, df=len(a)-1)
print(cv_simple_mean) #-1.6577592849346416

if tstat.statistic < cv_simple_mean:
	print("T Test: H0 is rejected under 0.05")
else:
	print("T Test: H0 is NOT rejected under 0.05")
	 

#3.2 p-value를 이용하는 방법
if tstat.pvalue < alpha:
	print("T Test with p-value: H0 is rejected under 0.05")
else:
	print("T-Test with p-value: H0 is NOT rejected under 0.05")
    
    
###############T검정(독립 소표본, 평균 비교)#############
#두 집단이 독립적일 때, 두 변수의 평균 차이에 대한 가설 검정을 수행하는 경우     
#자유도 df = n1 + n2 -2 
#검정통계량 계산 - scipy.stats.ttest_ind - 등분산: equal_var     
#독립된 집단 간 평균 비교는 T검정이 아닌 ANOVA 검정으로도 수행 가능
     # scipy.stats.f_oneway
            
import scipy
from scipy import stats

#1. 표본 평균(집단 간의 차이에 대한 평균 
#   -> 평균(처리집단 - 통제집단) = 평균(처리집단) - 평균(통제집단)
#   -> 평균의 차

mean_before = a['bp_before'].mean().round(2)
mean_after = a['bp_after'].mean().round(2)
mean_diff = mean_after - mean_before
print(mean_diff.round(2))  #-5.09


#2. 가설 검정 = 검정 통계량 or p-value
tstat_mean_diff = scipy.stats.ttest_ind(a['bp_after'], a['bp_before'], equal_var =True, alternative='less')
print(tstat_mean_diff) #Ttest_indResult(statistic=-3.0669836819036274, pvalue=0.0012061387390394456)
print("t-value", tstat_mean_diff.statistic.round(4))
print("p-value", tstat_mean_diff.pvalue.round(4))

#3. 가설 검정 - cv vs. pvalue
alpha= 0.05

cv_mean_diff = scipy.stats.t.ppf(q=alpha, df= 2*len(a)-2)
print(cv_mean_diff)

if tstat_mean_diff.statistic < cv_mean_diff:
	print("T-test: H0 is rejected")
else: 
	print("T-test: H0 is NOT rejected")
	
	
if tstat_mean_diff.pvalue < alpha:
	print("T-test with p-value: H0 is rejected")
else: 
	print("T-test with p-value: H0 is NOT rejected")
# T-test: H0 is rejected
# 치료전 집단과 치료 후 집단의 평균적인 차이가 같지 않고 치료 후 집단의 평균이 더 작다라는 대립가설을 채택
# 실제 시험에서는 그냥 값이 작은지 큰지만 판단하면 된다. 


#T-test => 2개의 집단 비교 / 3개 이상의 집단 => ANOVA
#하지만 2개의 집단의 평균을 비교하는데, 두개의 집단이 독립적이고, 분산이 동일한 평균의 차이를 test하면 ANOVA test도 가능하다.

#################F검정(독립 소표본, 평균 비교)####################
#독립 평균 비교는 T검정이 아닌 F검정(ANOVA)을 이용 가능
# 검정통계량 계산  scipy.stats.f_oneway

import scipy
from scipy import stats

#ANOVA Test
anova_test = scipy.stats.f_oneway(a['bp_after'], a['bp_before'])
print(anova_test)
#F_onewayResult(statistic=9.406388905063123, pvalue=0.002412277478078879)

#cv 
#dfn - 분자에 대한 자유도 (k- 1) , dfd - 분모에 대한 자유도 ( n -k)
alpha = 0.05
anova_cv = scipy.stats.f.ppf(q = 1-alpha, dfn = 2-1, dfd = len(a) -2)
print(anova_cv)

#Test
if anova_test.statistic > anova_cv:
	print("H0 is rejected under 0.05")
else:
	print("H0 is not rejected under 0.05")

if anova_test.pvalue < alpha:
	print("H0 is rejected under 0.05")
else:
	print("H0 is not rejected under 0.05")
    
    
####################연관된 두 집단, 쌍체 집단 T-TEST###############################
# 같은 사람의 전과 후 비교 -> 독립적인 집단이라고 보기 어렵다. 연관된 집단의 ttest
# => scipy.stats.ttest_rel (a,b, alternative="less") #two-sided #greater

import scipy
from scipy import stats

#1. 표본평균 E(a-b) = E(a)-E(b)
mean_before = a['bp_before'].mean().round(2)
mean_after = a['bp_after'].mean().round(2)
mean_diff = mean_after - mean_before
print(mean_diff.round(2))
#-5.09

#2. 검정통계량과 p-value 구하기
tstat_mean_diff = scipy.stats.ttest_rel(a['bp_after'],a['bp_before'], alternative="less" )
print(tstat_mean_diff)
print("t-value: ", tstat_mean_diff.statistic.round(4))
print("p-value: ", tstat_mean_diff.pvalue.round(4))

#3. 가설 검정
alpha = 0.05
if tstat_mean_diff.pvalue < alpha:
	print("H0 is rejected under 0.05")
else: print("H0 is NOT rejected under 0.05")

#Critical Value
cv_mean_diff = scipy.stats.t.ppf(q=alpha, df= 2*len(a) -2)
print(cv_mean_diff)
if tstat_mean_diff.statistic < cv_mean_diff:
	print("H0 is rejected under 0.05")
else: print("H0 is NOT rejected under 0.05")
    
    
    
###################카이제곱검정###############################
# 범주형 자료의 독립성 검정에 이용
#Contingency Table(분할표)는 범주형 자료들의 표본으로 재구성한 표
#범주 간의 독립 여부는 행의 확률과 열의 확률의 곱이 전체 표본에서
#분할표에 해당하는 값의 비율과 얼마나 차이 나는지로 검정 
#Contingency Table -> pandas.crosstab
#scipy.stats.chi2_contingency

import pandas as pd

a = pd.read_csv('data/blood_pressure.csv', index_col=0)


# 사용자 코딩
#귀무가설: independence
#대립가설: dependent
import scipy
from scipy import stats

#1. Contingency Table 
cont_table = pd.crosstab(a['sex'],a['agegrp'])
print(cont_table)


#2. Chi*2 test
from scipy.stats import chi2_contingency
#return 값은 첫번째가 statistic, 2. p-value , 3. df
chi2_test = chi2_contingency(cont_table)
print(chi2_test)
print("Test Statistic: ", chi2_test[0])
print("pvalue: ", chi2_test[1])
# Test Statistic:  0.0
# pvalue:  1.0

#Test
alpha = 0.05
if chi2_test[1] < alpha:
		print("H0 is rejected under 0.05: Two variables are dependent") 
		#귀무가설이 기각되면 의존적이라는 것을 의미
	else:
		print("H0 is NOT rejected under 0.05 : Two variables are independent")



        
##################윌콕슨 부호 검정#######################
#윌콕슨 검정은 두 집단의 분포에 대한 유사도를 검정하는 기법
#wilcoxon rank-sum => scipy.stats.ranksums
#wilcoxon signed-rank => scipy.stats.wilcoxon

#paired sample이 같은 분포로부터 도출(귀무가설)
#서로 다른 분포로부터 도출되었다. (대립가설)

#Non-parametric paired T-test라면
#H0 - same distribution
import scipy
from scipy import stats
from scipy.stats import wilcoxon

wilcoxon = scipy.stats.wilcoxon(a['bp_after'], a['bp_before'], alternative ='less')
print("Test statistic:", wilcoxon.statistic.round(4))
print("p-value:", wilcoxon.pvalue.round(4))
#> WilcoxonResult(statistic=2234.5, pvalue=0.0007053666782721429)

if wilcoxon.pvalue<0.05:
	print("H0 is rejected under 0.05: different distribution.")
else:
	print("H0 is NOT rejected under 0.05: same distribution.")
#귀무가설 기각 -> 치료전 분포와 치료 후 분포는 다른 분포로부터 도출되었다. 