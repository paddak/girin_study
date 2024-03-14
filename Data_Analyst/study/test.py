import numpy as np
import pandas as pd

#데이터 불러오기
ad=pd.read_csv('/Users/jiyul/Github/girin_study/Data_Analyst/study/자료/ad1.csv')
log=pd.read_csv('/Users/jiyul/Github/girin_study/Data_Analyst/study/자료/log1.csv')
od=pd.read_csv('/Users/jiyul/Github/girin_study/Data_Analyst/study/자료/order1.csv')

#데이터 전처리 자동화
#목적 : 주 단위로 유입경로별 성과를 트레킹
#CPC: cost per click
#ROAS (Return On Advertising Spend)
#정의 : 광고 지출에 대한 수익을 나카내는 마케팅 지표
#계산벙법 : 광고로 인한 수익/광고지출
#높은 ROAS: 광고 지출 대비 높은 수익의미. 광고 캠페인 효과성 나타냄
#낮은 ROAS: 광고 투자에 대한 낮은 수익 의미. 광고 전략 개선 필요


ad['date']=pd.to_datetime(ad.date)
#date열 dtype을 datetime으로 변환

#주 단위로 데이터 집계하기 위해 변환
ad['week']=ad.date.dt.to_period("W").dt.to_timestamp()
#주 단위로 변환후 그 주에 시작하는 날짜로 변환한걸 새로윤 week열 만들어 추가

#캠페인이 week단위로 click이 몇번 발생했고, spend이 얼마나 발생했는지 합산 해주는 집계
ad_agg = ad.groupby(by=['campaign','week'])[['spend','clicks']].sum().reset_index()

#cpc계산
ad_agg['cpc']=ad_agg.spend/ad_agg.clicks

log['timestamp']=pd.to_datetime(log.timestamp)

#campaign정보 없는 것들 필터링
log2=log[~log.campaign.isna()]

od['order_date']=pd.to_datetime(od.order_date)


log.groupby('order_id').size().sort_values()
od.groupby('order_id').size().sort_values()

#od와 log 병합
od1=pd.merge(od,log,how='left',left_on='order_id',right_on='order_id')

#nan값 organic으로 변환
od1['campaign']=od1.campaign.fillna('organic')

od1['week']=od1.order_date.dt.to_period('W').dt.to_timestamp()
od_agg=od1.groupby(by=['campaign','week'])['order_amount'].sum().reset_index()
od1