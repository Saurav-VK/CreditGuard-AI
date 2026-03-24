#!/usr/bin/env python
# coding: utf-8

# In[6]:


from sklearn.base import BaseEstimator , TransformerMixin
import numpy as np
import pandas as pd


# CUSTOM TRANSFORMER FOR REMOVING ROWS

# In[7]:


class ColumnDropper(BaseEstimator , TransformerMixin):
    def __init__(self , cols):
        self.cols = cols
    def fit(self , X , y = None):
        return self
    def transform(self , X):
        X = X.copy()
        return X.drop(columns = self.cols , errors = 'ignore')


# NULL ROW REMOVER

# In[8]:


class NullRowRemover(BaseEstimator , TransformerMixin):
    def __init__(self , cols):
        self.cols = cols
    def fit(self , X , y = None):
        return self
    def transform(self , X):
        X = X.copy()
        return X[X[self.cols].notnull().all(axis = 1)]


# BINARY MAPPER

# In[9]:


class BinaryMapper(BaseEstimator , TransformerMixin):
    def __init__(self , mapping_dict):
        self.mapping_dict = mapping_dict
    def fit(self , X , y = None):
        return self
    def transform(self , X):
        X = X.copy()
        for col , mapping in self.mapping_dict.items():
            X[col] = X[col].map(mapping)
        return X


# COLUMN FLAGGER

# In[10]:


class ColumnFlagger(BaseEstimator , TransformerMixin):
    def __init__(self , cols):
        self.cols = cols
    def fit(self , X , y = None):
        return self
    def transform(self , X):
        X = X.copy()
        for col in self.cols:
            X[f'NO_{col}_FLAG'] = X[col].isnull().astype(int)
            X[col] = X[col].fillna(0)
        return X


# FILLNA(0)

# In[11]:


class Filler(BaseEstimator , TransformerMixin):
    def __init__(self , cols):
        self.cols = cols
    def fit(self , X , y = None):
        return self
    def transform(self , X):
        X = X.copy()
        X[self.cols] = X[self.cols].fillna(0)
        return X
    


# In[12]:


class StringFiller(BaseEstimator , TransformerMixin):
    def __init__(self , cols):
        self.cols = cols
    def fit(self , X , y = None):
        return self
    def transform(self , X):
        X = X.copy()
        X[self.cols] = X[self.cols].fillna('Unknown')
        return X
    


# FLAGGING AND FILLING

# In[13]:


class FlagAndFill(BaseEstimator , TransformerMixin):
    def __init__(self , cols):
        self.cols = cols
    def fit(self , X , y = None):
        return self
    def transform(self , X):
        X = X.copy()
        existing_cols = [col for col in self.cols if col in X.columns]
        X['no_comm_flag'] = X[existing_cols].isnull().all(axis = 1).astype(int)
        X[existing_cols] = X[existing_cols].fillna(0)
        return X


# FREQUENCY ENCODING

# In[14]:


class FrequencyEncoder(BaseEstimator , TransformerMixin):
    def __init__(self , cols):
        self.cols = cols
        self.freq_dict = {}
    def fit(self , X , y = None):
        for col in self.cols:
            self.freq_dict[col] = X[col].value_counts().to_dict()
        return self
    def transform(self , X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.freq_dict[col]).fillna(0)
        return X


# ONE HOT ENCODING

# In[15]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class Encoder(BaseEstimator , TransformerMixin):
    def __init__(self , cols):
        self.cols = cols
        
        self.encoder = OneHotEncoder(drop= 'first' , handle_unknown = 'ignore' , sparse_output = False)
    def fit(self , X , y = None):
        self.encoder.fit(X[self.cols])
        self.feature_names = self.encoder.get_feature_names_out(self.cols)
        return self
    def transform(self , X):
        X = X.copy()
        encoded = self.encoder.transform(X[self.cols])
        encoded_df = pd.DataFrame(data = encoded , index = X.index , columns = self.feature_names)
        X = X.drop(columns = self.cols)
        X = pd.concat([X , encoded_df] , axis = 1)
        return X


# REMOVING THE ADDITIONAL COLUMN

# In[16]:


class RemoveCol(BaseEstimator , TransformerMixin):
    def __init__(self , cols):
        self.cols = cols
    def fit(self , X , y = None):
        return self
    def transform(self , X):
        X = X.copy()
        X = X.drop(columns = self.cols , errors = 'ignore')
        return X
        


# FEATURE ENGINEERING

# In[17]:


class AddCols(BaseEstimator , TransformerMixin):
    def __init__(self , cols):
        self.cols = cols
    def fit(self , X , y = None):
        return self
    def transform(self , X):
        X = X.copy()
        X['Loan_to_income_Ratio'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
        X['Annuity_to_income_Ratio'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
        X['Goods_to_Credit_Ratio']  = X['AMT_GOODS_PRICE'] / X['AMT_CREDIT']
        # X['Employment Range'] =  X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
        X['EXT_SOURCE_MEAN'] = (X['EXT_SOURCE_1'] + X['EXT_SOURCE_2'] +  X['EXT_SOURCE_3']) / 3
        X['MIN_EXT_SOURCE'] = X[['EXT_SOURCE_1' , 'EXT_SOURCE_2' , 'EXT_SOURCE_3']].min(axis = 1)
        X['MAX_EXT_SOURCE'] = X[['EXT_SOURCE_1' , 'EXT_SOURCE_2' , 'EXT_SOURCE_3']].max(axis = 1)
        X['NO_EMPLOYMENT_DETAILS_FLAG'] = np.where(X['DAYS_EMPLOYED'] == 365243 , 1 , 0)
        X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace(365243 , 0)
        X['Employment Range'] =  X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
        X['INCOME_TO_CHILDREN'] = X['AMT_INCOME_TOTAL'] / X['CNT_CHILDREN']
        X['ANNUITY_TO_FAMILTY_MEMBERS'] = X['AMT_ANNUITY'] / X['CNT_FAM_MEMBERS']
        X['AGE'] = (X['DAYS_BIRTH'] * -1) / 365
        X['NO_CHILDREN_FLAG'] = np.where(X['CNT_CHILDREN'] == 0 , 1 , 0)
        X['INCOME_TO_CHILDREN'] = X['INCOME_TO_CHILDREN'].replace([np.inf , -np.inf] , 0)
        X['MEAN_TIMES_LTI'] = X['EXT_SOURCE_MEAN'] * X['Loan_to_income_Ratio']
        X['AGE_TIMES_LTI'] = X['AGE'] * X['Loan_to_income_Ratio']
        X['Employment_Years'] = X['DAYS_EMPLOYED'] / 365
        X['Employment_TIMES_LTI'] = X['Employment_Years'] * X['Loan_to_income_Ratio']
        X['TOTAL_BUREAU_REQUESTS'] = X['AMT_REQ_CREDIT_BUREAU_HOUR'] + X['AMT_REQ_CREDIT_BUREAU_DAY'] +X['AMT_REQ_CREDIT_BUREAU_WEEK'] + X['AMT_REQ_CREDIT_BUREAU_MON'] + X['AMT_REQ_CREDIT_BUREAU_YEAR']
        X['RECENT_BUREAU_REQ'] = X['AMT_REQ_CREDIT_BUREAU_HOUR'] + X['AMT_REQ_CREDIT_BUREAU_DAY'] + X['AMT_REQ_CREDIT_BUREAU_WEEK']
        X['LONG_TERM_BUREAU_REQUESTS'] = X['AMT_REQ_CREDIT_BUREAU_MON'] + X['AMT_REQ_CREDIT_BUREAU_YEAR']
        return X


# COLUMN ORDERING

# In[29]:


class ColumnOrder(BaseEstimator , TransformerMixin):
    def __init__(self , cols):
        self.cols = cols
    def fit(self , X , y = None):
        return self
    def transform(self , X):
        X = X.copy()
        X = X[self.cols]
        return X


# In[19]:


from joblib import load
order_cols = load('ColumnOrder.pkl')


# In[24]:


mapping_dict = {'CODE_GENDER' : {'M' : 1 , 'F' : 0 , 'XNA' : 2} , 'FLAG_OWN_CAR' : {'Y' : 1 , 'N' : 0} , 'FLAG_OWN_REALTY' : {'Y' : 1 , 'N' : 0} , 'WEEKDAY_APPR_PROCESS_START' : {'SUNDAY' : 1 , 'MONDAY' : 2 , 'TUESDAY' : 3 , 'WEDNESDAY' : 4 , 'THURSDAY' : 5 , 'FRIDAY' : 6 , 'SATURDAY' : 7}}
flag_columns = ['AMT_REQ_CREDIT_BUREAU_HOUR' , 'AMT_REQ_CREDIT_BUREAU_DAY' , 'AMT_REQ_CREDIT_BUREAU_WEEK' , 'AMT_REQ_CREDIT_BUREAU_MON' , 'AMT_REQ_CREDIT_BUREAU_QRT' , 'AMT_REQ_CREDIT_BUREAU_YEAR']
filler_cols = ['DEF_60_CNT_SOCIAL_CIRCLE' , 'DEF_30_CNT_SOCIAL_CIRCLE' , 'OBS_30_CNT_SOCIAL_CIRCLE' , 'OBS_60_CNT_SOCIAL_CIRCLE' , 'AMT_GOODS_PRICE']
drop_cols = ['TARGET' , 'YEARS_BEGINEXPLUATATION_MODE' , 'YEARS_BEGINEXPLUATATION_MEDI' , 'YEARS_BEGINEXPLUATATION_AVG' , 'EMERGENCYSTATE_MODE', 'SK_ID_CURR' ,
 'TOTALAREA_MODE',
 'FLOORSMAX_MEDI',
 'FLOORSMAX_AVG',
 'FLOORSMAX_MODE',
 'HOUSETYPE_MODE',
 'LIVINGAREA_MEDI',
 'LIVINGAREA_MODE',
 'LIVINGAREA_AVG',
 'ENTRANCES_AVG',
 'ENTRANCES_MEDI',
 'ENTRANCES_MODE',
 'APARTMENTS_MEDI',
 'APARTMENTS_AVG',
 'APARTMENTS_MODE',
 'WALLSMATERIAL_MODE',
 'ELEVATORS_MODE',
 'ELEVATORS_AVG',
 'ELEVATORS_MEDI',
 'NONLIVINGAREA_MEDI',
 'NONLIVINGAREA_AVG',
 'NONLIVINGAREA_MODE',
 'BASEMENTAREA_MODE',
 'BASEMENTAREA_AVG',
 'BASEMENTAREA_MEDI',
 'LANDAREA_MEDI',
 'LANDAREA_MODE',
 'LANDAREA_AVG',
 'YEARS_BUILD_MEDI',
 'YEARS_BUILD_MODE',
 'YEARS_BUILD_AVG',
 'FLOORSMIN_AVG',
 'FLOORSMIN_MODE',
 'FLOORSMIN_MEDI',
 'LIVINGAPARTMENTS_MEDI',
 'LIVINGAPARTMENTS_MODE',
 'LIVINGAPARTMENTS_AVG',
 'FONDKAPREMONT_MODE',
 'NONLIVINGAPARTMENTS_MEDI',
 'NONLIVINGAPARTMENTS_AVG',
 'NONLIVINGAPARTMENTS_MODE',
 'COMMONAREA_AVG',
 'COMMONAREA_MODE',
 'COMMONAREA_MEDI']
column_flag_cols = ['EXT_SOURCE_1' , 'EXT_SOURCE_2' , 'EXT_SOURCE_3' ,'OWN_CAR_AGE']
encoding_cols = ['NAME_CONTRACT_TYPE' , 'NAME_TYPE_SUITE' , 'NAME_INCOME_TYPE' , 'NAME_EDUCATION_TYPE' , 'NAME_FAMILY_STATUS' , 'NAME_HOUSING_TYPE' , 'OCCUPATION_TYPE']
string_cols = ['NAME_TYPE_SUITE']
null_row_cols = ['DAYS_LAST_PHONE_CHANGE' , 'CNT_FAM_MEMBERS' , 'CODE_GENDER' , 'AMT_ANNUITY']
freq_cols = ['ORGANIZATION_TYPE']
additional = ['OCCUPATION_TYPE_nan']


# In[30]:


from sklearn.pipeline import Pipeline

pipeline = Pipeline(steps=[

    ('remove_null_rows', NullRowRemover(cols=null_row_cols)),

    ('binary_mapping', BinaryMapper(mapping_dict=mapping_dict)),

    ('fill_numeric_nulls', Filler(cols=filler_cols)),

    ('flag_and_fill_comm', FlagAndFill(cols=flag_columns)),

    ('column_flagging', ColumnFlagger(cols=column_flag_cols)),

    ('drop_columns', ColumnDropper(cols=drop_cols)),

    ('fill_string_nulls', StringFiller(cols=string_cols)),

    ('frequency_encoding', FrequencyEncoder(cols=freq_cols)),

    ('one_hot_encoding', Encoder(cols=encoding_cols)),

    ('removing the additional column arising from OHE' , RemoveCol(cols = additional)),

    ('Feature Engineering' , AddCols(cols = [])),

    ('Ordering Columns' , ColumnOrder(cols = order_cols))

])

                                 


# In[ ]:




