#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


application_df=pd.read_csv(r'C:\Users\ASUS\Desktop\priority list mini project\PROJECT\application_data.csv')


# In[3]:


previous_df=pd.read_csv(r'C:\Users\ASUS\Desktop\priority list mini project\PROJECT\previous_application.csv')


# In[4]:


application_nulldf=application_df.isna().mean().round(4)*100


# In[5]:


nullapp_40_features=application_nulldf[application_nulldf>=40].index


# In[6]:


previous_nulldf=previous_df.isna().mean().round(4)*100


# In[7]:


nullprev_40_features=previous_nulldf[previous_nulldf>=40].index


# In[8]:


source = application_df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','TARGET']]


# In[9]:


source.drop(['EXT_SOURCE_1','TARGET'],axis=1,inplace=True)


# In[10]:


flag_col_doc=['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6','FLAG_DOCUMENT_7', 
           'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
           'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
           'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
df_flag=application_df[flag_col_doc+['TARGET']]
df_flag["TARGET"] = df_flag["TARGET"].replace({1:"Defaulter",0:"Repayer"})
length = len(flag_col_doc)


# In[11]:


df_flag.drop(['FLAG_DOCUMENT_3','TARGET','FLAG_DOCUMENT_6','FLAG_DOCUMENT_8'],axis=1,inplace=True)


# In[12]:


contact_col=application_df[['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
              'FLAG_PHONE', 'FLAG_EMAIL','TARGET']]


# In[13]:


contact_col.drop(['TARGET'],axis=1,inplace=True)


# In[14]:


application_df.drop(nullapp_40_features,axis=1,inplace=True)


# In[15]:


unwanted_col=source+contact_col+df_flag


# In[16]:


application_df.drop(unwanted_col,axis=1,inplace=True)


# In[17]:


previous_df.drop(nullprev_40_features,axis=1,inplace=True)


# In[18]:


previous_df.drop(['WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START',
                        'FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY'],axis=1,inplace=True)


# In[19]:


application_df['NAME_TYPE_SUITE'].fillna((application_df['NAME_TYPE_SUITE'].mode()[0]),inplace = True)


# In[20]:


application_df['OCCUPATION_TYPE'].fillna('UNKNOWN',inplace=True)


# In[21]:


amount = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
         'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']

for col in amount:
    application_df[col].fillna(application_df[col].median(),inplace = True)


# In[22]:


application_df['AMT_GOODS_PRICE'].fillna(application_df['AMT_GOODS_PRICE'].median(),inplace=True)


# In[23]:


social_circle=['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
                'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']
for social in social_circle:
    application_df[social].fillna(application_df[social].median(),inplace=True)


# In[24]:


application_df=application_df.dropna()


# In[25]:


previous_df['PRODUCT_COMBINATION'].fillna((previous_df['PRODUCT_COMBINATION'].mode()[0]),inplace = True)


# In[26]:


amount = ['AMT_ANNUITY','AMT_GOODS_PRICE']

for col in amount:
    previous_df[col].fillna(previous_df[col].median(),inplace = True)


# In[27]:


previous_df.loc[previous_df['CNT_PAYMENT'].isnull(),'NAME_CONTRACT_STATUS'].value_counts()


# In[28]:


previous_df['CNT_PAYMENT'].fillna(0,inplace = True)


# In[29]:


previous_df['AMT_CREDIT'].isnull().sum()


# In[30]:


previous_df=previous_df.dropna()


# In[31]:


application_numeric_col=application_df[['SK_ID_CURR', 'TARGET', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
       'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
       'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
       'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'CNT_FAM_MEMBERS',
       'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
       'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION',
       'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
       'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
       'LIVE_CITY_NOT_WORK_CITY', 'OBS_30_CNT_SOCIAL_CIRCLE',
       'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
       'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_3',
       'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_8', 'AMT_REQ_CREDIT_BUREAU_HOUR',
       'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
       'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
       'AMT_REQ_CREDIT_BUREAU_YEAR']]


# In[32]:


application_df['DAYS_BIRTH'] = abs(application_df['DAYS_BIRTH'])
application_df['DAYS_EMPLOYED'] = abs(application_df['DAYS_EMPLOYED'])
application_df['DAYS_ID_PUBLISH'] = abs(application_df['DAYS_ID_PUBLISH'])
application_df['DAYS_LAST_PHONE_CHANGE'] = abs(application_df['DAYS_LAST_PHONE_CHANGE'])
application_df['DAYS_REGISTRATION'] = abs(application_df['DAYS_REGISTRATION'])


# In[33]:


previous_numeric_col=previous_df[['SK_ID_PREV', 'SK_ID_CURR', 'AMT_ANNUITY', 'AMT_APPLICATION',
       'AMT_CREDIT', 'AMT_GOODS_PRICE', 'DAYS_DECISION', 'SELLERPLACE_AREA',
       'CNT_PAYMENT']]


# In[34]:


previous_df['DAYS_DECISION'] = abs(previous_df['DAYS_DECISION'])
previous_df['SELLERPLACE_AREA'] = abs(previous_df['SELLERPLACE_AREA'])


# In[35]:


uv = np.percentile(application_df.CNT_CHILDREN,[99])[0]
application_df[(application_df.CNT_CHILDREN>uv)]
application_df.CNT_CHILDREN[(application_df.CNT_CHILDREN>3*uv)]=3*uv


# In[36]:


uv = np.percentile(application_df.AMT_ANNUITY,[99])[0]
application_df[(application_df.AMT_ANNUITY>uv)]
application_df.AMT_ANNUITY[(application_df.AMT_ANNUITY>3*uv)]=3*uv


# In[37]:


uv = np.percentile(application_df.AMT_CREDIT,[99])[0]
application_df[(application_df.AMT_CREDIT>uv)]
application_df.AMT_CREDIT[(application_df.AMT_CREDIT>3*uv)]=3*uv


# In[38]:


uv = np.percentile(application_df.AMT_GOODS_PRICE,[99])[0]
application_df[(application_df.AMT_GOODS_PRICE>uv)]
application_df.AMT_GOODS_PRICE[(application_df.AMT_GOODS_PRICE>3*uv)]=3*uv


# In[39]:


uv = np.percentile(application_df.AMT_REQ_CREDIT_BUREAU_YEAR,[99])[0]
application_df[(application_df.AMT_REQ_CREDIT_BUREAU_YEAR>uv)]
application_df.AMT_REQ_CREDIT_BUREAU_YEAR[(application_df.AMT_REQ_CREDIT_BUREAU_YEAR>3*uv)]=3*uv


# In[40]:


uv = np.percentile(application_df.DAYS_REGISTRATION,[99])[0]
application_df[(application_df.DAYS_REGISTRATION>uv)]
application_df.DAYS_REGISTRATION[(application_df.DAYS_REGISTRATION>3*uv)]=3*uv


# In[41]:


uv = np.percentile(application_df.CNT_FAM_MEMBERS,[99])[0]
application_df[(application_df.CNT_FAM_MEMBERS>uv)]
application_df.CNT_FAM_MEMBERS[(application_df.CNT_FAM_MEMBERS>3*uv)]=3*uv


# In[42]:


uv = np.percentile(previous_df.AMT_ANNUITY,[99])[0]
previous_df[(previous_df.AMT_ANNUITY>uv)]
previous_df.AMT_ANNUITY[(previous_df.AMT_ANNUITY>3*uv)]=3*uv


# In[43]:


uv = np.percentile(previous_df.AMT_APPLICATION,[99])[0]
previous_df[(previous_df.AMT_APPLICATION>uv)]
previous_df.AMT_APPLICATION[(previous_df.AMT_APPLICATION>3*uv)]=3*uv


# In[44]:


uv = np.percentile(previous_df.AMT_CREDIT,[99])[0]
previous_df[(previous_df.AMT_CREDIT>uv)]
previous_df.AMT_CREDIT[(previous_df.AMT_CREDIT>3*uv)]=3*uv


# In[45]:


uv = np.percentile(previous_df.AMT_GOODS_PRICE,[99])[0]
previous_df[(previous_df.AMT_GOODS_PRICE>uv)]
previous_df.AMT_GOODS_PRICE[(previous_df.AMT_GOODS_PRICE>3*uv)]=3*uv


# In[46]:


uv = np.percentile(previous_df.CNT_PAYMENT,[99])[0]
previous_df[(previous_df.CNT_PAYMENT>uv)]
previous_df.CNT_PAYMENT[(previous_df.CNT_PAYMENT>3*uv)]=3*uv


# In[47]:


uv = np.percentile(previous_df.DAYS_DECISION,[99])[0]
previous_df[(previous_df.DAYS_DECISION>uv)]
previous_df.DAYS_DECISION[(previous_df.DAYS_DECISION>3*uv)]=3*uv


# In[48]:


uv = np.percentile(previous_df.SELLERPLACE_AREA,[99])[0]
previous_df[(previous_df.SELLERPLACE_AREA>uv)]
previous_df.SELLERPLACE_AREA[(previous_df.SELLERPLACE_AREA>3*uv)]=3*uv


# In[49]:


loan_process_df = pd.merge(application_df, previous_df, how='inner', on='SK_ID_CURR')


# In[50]:


loan_process_df_cat=loan_process_df[['NAME_CONTRACT_TYPE_x', 'CODE_GENDER', 'FLAG_OWN_CAR',
       'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
       'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
       'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
       'NAME_CONTRACT_TYPE_y', 'NAME_CASH_LOAN_PURPOSE',
       'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON',
       'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO',
       'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY',
       'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION']]


# In[51]:


import category_encoders as ce


# In[52]:


encoder= ce.TargetEncoder()
data_encoder=encoder.fit_transform(loan_process_df_cat,loan_process_df['TARGET'])


# In[53]:


loan_process_df.drop(['NAME_CONTRACT_TYPE_x', 'CODE_GENDER', 'FLAG_OWN_CAR',
       'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
       'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
       'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
       'NAME_CONTRACT_TYPE_y', 'NAME_CASH_LOAN_PURPOSE',
       'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON',
       'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO',
       'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY',
       'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION'],axis=1,inplace=True)


# In[54]:


loan_process_df=pd.concat([loan_process_df,data_encoder],axis=1)


# In[55]:


pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 400)
loan_process_df


# In[56]:


#Check the multicollinearity in the dataset
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[57]:


Z= loan_process_df.drop(['TARGET','OBS_30_CNT_SOCIAL_CIRCLE','FLAG_OWN_REALTY','NAME_TYPE_SUITE',
                        'NAME_PORTFOLIO','WEEKDAY_APPR_PROCESS_START','FLAG_OWN_CAR','NAME_GOODS_CATEGORY',
                        'NAME_CONTRACT_TYPE_x','NAME_PAYMENT_TYPE','NAME_CLIENT_TYPE','NAME_CONTRACT_TYPE_y',
                        'REGION_RATING_CLIENT','NAME_CASH_LOAN_PURPOSE','NAME_YIELD_GROUP','NAME_SELLER_INDUSTRY',
                        'AMT_CREDIT_x','AMT_GOODS_PRICE_y','NAME_CONTRACT_STATUS','CHANNEL_TYPE','NAME_PRODUCT_TYPE',
                        'NAME_HOUSING_TYPE','NAME_FAMILY_STATUS','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
                        'CODE_GENDER','ORGANIZATION_TYPE','CNT_FAM_MEMBERS','AMT_CREDIT_y','PRODUCT_COMBINATION',
                        'DAYS_BIRTH','CODE_REJECT_REASON','OCCUPATION_TYPE','SK_ID_PREV','REGION_RATING_CLIENT_W_CITY',
                        'HOUR_APPR_PROCESS_START','AMT_ANNUITY_x','REG_REGION_NOT_WORK_REGION'],axis=1)
  
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = Z.columns 
  
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(Z.values, i) 
                          for i in range(len(Z.columns))] 
print(vif_data)


# In[58]:


loan_process_df1=loan_process_df.drop(['OBS_30_CNT_SOCIAL_CIRCLE','FLAG_OWN_REALTY','NAME_TYPE_SUITE',
                        'NAME_PORTFOLIO','WEEKDAY_APPR_PROCESS_START','FLAG_OWN_CAR','NAME_GOODS_CATEGORY',
                        'NAME_CONTRACT_TYPE_x','NAME_PAYMENT_TYPE','NAME_CLIENT_TYPE','NAME_CONTRACT_TYPE_y',
                        'REGION_RATING_CLIENT','NAME_CASH_LOAN_PURPOSE','NAME_YIELD_GROUP','NAME_SELLER_INDUSTRY',
                        'AMT_CREDIT_x','AMT_GOODS_PRICE_y','NAME_CONTRACT_STATUS','CHANNEL_TYPE','NAME_PRODUCT_TYPE',
                        'NAME_HOUSING_TYPE','NAME_FAMILY_STATUS','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
                        'CODE_GENDER','ORGANIZATION_TYPE','CNT_FAM_MEMBERS','AMT_CREDIT_y','PRODUCT_COMBINATION',
                        'DAYS_BIRTH','CODE_REJECT_REASON','OCCUPATION_TYPE','SK_ID_PREV','REGION_RATING_CLIENT_W_CITY',
                        'HOUR_APPR_PROCESS_START','AMT_ANNUITY_x','REG_REGION_NOT_WORK_REGION'],axis=1)
  


# In[59]:


loan_process_df1=loan_process_df1.sample(n =10000) 


# In[60]:


X=loan_process_df1.drop(['TARGET'],axis=1)
y=loan_process_df1['TARGET']


# In[61]:


from sklearn import preprocessing
standardisation=preprocessing.StandardScaler()
X=standardisation.fit_transform(X)


# In[62]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.3)


# In[63]:


from sklearn.ensemble import RandomForestClassifier


# In[64]:


from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score


# In[65]:



rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)


# In[66]:


print('Random forest classifier of accuracy score is:',accuracy_score(y_test,y_pred))
print('Random forest classifier of precision score is:',precision_score(y_test,y_pred))
print('Random forest classifier  of recall score is:',recall_score(y_test,y_pred))
print('Random forest classifier  of F1 score is:',f1_score(y_test,y_pred))
confusion_matrix(y_test,y_pred)


# In[67]:


import pickle


# In[68]:



#pickle.dump(data_encoder,open('encoder.pkl','wb')) 
pickle.dump(rfc,open('model.pkl','wb'))


# In[ ]:




