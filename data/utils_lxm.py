from datetime import datetime 
import pandas as pd
import numpy as np
import os, sys

from pandas.tseries.holiday import Holiday, USMemorialDay, AbstractHolidayCalendar, nearest_workday, MO



# ----------------- Transformação dos dados crus em um novo dataset (raw data) -----------------

def dataset_transform(df):
    '''
    Transforma a base original em uma visão: [Data, Estação, , Número de pessoas que entraram]
    
    Args:
    
    Return: 
    
    '''
    df['date'] = df.time.str.split(" ").str[0]
    
    # criação de chave de catracas distintas
    entries_day_scp = df.groupby(['station', 'ca','scp', 'linename', 'date'], as_index = False).agg({'entries':'max'})
    entries_day_scp['station_linename_ca_scp'] = entries_day_scp['linename'] +entries_day_scp['station'] +entries_day_scp['ca'] +entries_day_scp['scp'] 
    
    # n_entries é o número de pessoas que passou por aquela catraca naquele dia.
    grouped = entries_day_scp.groupby('station_linename_ca_scp')
    entries_day_scp['n_entries'] = (entries_day_scp['entries'] - grouped['entries'].shift(1))
   
    # daily_entries é a correção do n_entries com acertos de nulos e números negativos
    # caudados pela recontagem das catracas
    entries_day_scp['daily_entries'] = entries_day_scp['n_entries']
    entries_day_scp.loc[entries_day_scp.daily_entries < 0, 'daily_entries'] =entries_day_scp.loc[entries_day_scp.n_entries < 0, 'entries']
    entries_day_scp['daily_entries'].fillna(0,inplace = True)
    entries_day_scp.drop('n_entries',axis = 1, inplace = True)
    
    # Somando total de pessoas que entraram nas catracas por estação
    entries_day = entries_day_scp.groupby(['station', 'date'], as_index = False).agg({'daily_entries':'sum'})

    year = df['date'][0][:4]
    entries_day.to_csv("df_tranformed_"+year+".csv", index = False)
    return entries_day






# ----------------- Situação das variáveis numéricas -----------------

#Returns null values (%)
def get_nans(df):
    nan_dic = {}
    for col in df.columns:
        if df[col].isnull().any() == True:
            nan_dic[col] = df[col].isnull().sum()
    return pd.DataFrame({
        'Feature': list(nan_dic.keys()),
        'Nulls': list(nan_dic.values()),
        'Percent': np.round((np.array(list(nan_dic.values())) / df.shape[0])*100, decimals = 1)
    }).sort_values('Nulls',ascending = False)

#Returns 0 values (%)
def get_zeros(df):
    zero_dic = {}
    for col in df.columns:
        if (df[col] == 0).sum() > 0:
            zero_dic[col] = (df[col] == 0).sum()
    return pd.DataFrame({'Feature': list(zero_dic.keys()),
                        'Zeros': list(zero_dic.values()),
                        'Percent': np.round((np.array(list(zero_dic.values())) / df.shape[0])*100, decimals = 1)
    }).sort_values('Zeros',ascending = False)


# ----------------- Transformação de valores categóricos -----------------

#Cleaner

class Cleaner:
  
    from sklearn.preprocessing import MinMaxScaler
  
    def __init__(self, dataframe, bin_list = [], scaler_list = [], ordinal_list = [], dummy_list = [], clean_nan = False):    
        self.dataframe = dataframe
        self.bin_list = bin_list
        self.scaler_list = scaler_list
        self.ordinal_list = ordinal_list
        self.dummy_list = dummy_list
        self.clean_nan = clean_nan
    
    def transform(self):    
        dataframe = self.dataframe
        dataframe_cols = dataframe.columns.values.tolist()

        #Binariza
        if bool(self.bin_list):
            for col in self.bin_list:
                dataframe[col] = [0 if x == 0 else 1 for x in dataframe[col]]

        #Escalona(MinMax)
        if bool(self.scaler_list):
            scaler = MinMaxScaler()
            dataframe.loc[:,self.scaler_list] = scaler.fit_transform(dataframe.loc[:,self.scaler_list])

#        #Label Ordinal  
        #Personalizar para dar o devido grau de importância para as categorias
#        if bool(self.ordinal_list):
#            for col in self.ordinal_list:
#                dataframe[col] = dataframe[col].map({'M':13, 'K':12, 'L':11, 'J':10, 'I':9, 'H':8, 'G':7,
#                                                     'F':6, 'E':5, 'D':4, 'C':3, 'B':2, 'A':1, np.nan:0})

        if bool(self.dummy_list):
            dataframe = pd.get_dummies(dataframe, columns=self.dummy_list, drop_first=True)

        if self.clean_nan:
            for col in dataframe.select_dtypes(exclude='object').columns.tolist():
                dataframe[col].fillna(dataframe[col].median(), inplace = True)

      
        return dataframe

    
# ----------------- Data Prep -----------------

def clean_date(df, col):
    results = []
    for element in df[col]:
        try:
            results.append(pd.to_datetime(element))
        except:
            results.append(np.nan)
    df[col] = results
    return df


def scale_df_cols(df, target_cols, scale_type='minmax'):
    df = df.copy()
    for col in target_cols:
        if(scale_type == 'minmax'):
            scaler = MinMaxScaler()
            X = scaler.fit_transform(np.array(df[col]).reshape(-1,1))
            df['norm_{}'.format(col)] = X
        elif(scale_type == 'standard'):
            scaler = StandardScaler()
            X = scaler.fit_transform(np.array(df[col]).reshape(-1,1))
            df['std_{}'.format(col)] = X
        elif(scale_type == 'log'):
            df['log_{}'.format(col)] = df[col].apply(np.log)
        elif(scale_type == 'gaussian_rank'):
            scaler = QuantileTransformer()
            X = scaler.fit_transform(np.array(df[col]).reshape(-1,1))
            df['qtd_{}'.format(col)] = X
    return df

# ----------------- Feature Engineering  -----------------


def clean_dataframe(df):
    new_df = df.copy()
    for col in list(new_df.columns.values):
        if(col != "@timestamp"):
            new_df[col] = new_df[col].apply(lambda x:  re.sub(r'[\[\]]',' ', str(x)))
            new_df[col] = new_df[col].apply(lambda x: re.sub(r"[\']",'', x))
    return new_df


def get_mothers_day(year):
    return pd.to_datetime(pytime.mother(year))


def span_year_range(year_1, year_2):
    low_bound = '{}-01-01'.format(year_1)
    high_bound = '{}-12-31'.format(year_2)
    return low_bound, high_bound
 

def is_weekday(df):
    results = []
    weekdays = set([0,1,2,3,4])
    for date in df.index.values:
        if(date.weekday() in weekdays):
            results.append(1)
        else:
            results.append(0)
    df['is_business_day'] = results
    return df

def is_business_day(df):
    results = []
    weekdays = set([0,1,2,3,4])
    for date in df['day_of_week']:
        if(date in weekdays):
            results.append(1)
        else:
            results.append(0)
    return results


def is_national_holiday(df, national_holidays):
    results = []
    national_set = set(national_holidays)
    for date in pd.to_datetime(df.index.values):
        if(date in national_set):
            results.append(1)
        else:
            results.append(0)
    return results


class ExampleCalendar(AbstractHolidayCalendar):
    rules = [
        USMemorialDay,
        Holiday('July 4th', month=7, day=4, observance=nearest_workday),
        Holiday('Columbus Day', month=10, day=1,
                offset=pd.DateOffset(weekday=MO(2)))]
    

# ----------------- Statistics  -----------------


def calc_horizontal_statistics(df, cols):
    df = df.copy()
    df['mean_across'] = df[cols].iloc[:,].mean(axis=1)
    df['median_across'] = df[cols].iloc[:,].median(axis=1)
    df['std_across'] = df[cols].iloc[:,].std(axis=1)
    return df
            
def gaussian_rank(df, cols):
    pass

def check_if_normal(df, col, test_type='normal'):
    # Usando o teste de Shapiro-wilk
    alpha = 0.05 # nível de significância
    to_test = np.array(df[col].values)
    
    # teste de shapiro retorna (estatística de teste, p valor)
    if(test_type == 'shapiro'):
        test_statistic, p_value = sct.shapiro(to_test)
    elif(test_type == 'jarque'):
        test_statistic, p_value = sct.jarque_bera(to_test)
    else:
        test_statistic, p_value = sct.normaltest(to_test)

    print('W =%.3f\np_value = %.3f'%(test_statistic, p_value))
    if(p_value > alpha):
        # Falhamos em rejeitar a H_O, sugerindo que a mostra pode ter vindo de uma população normalmente distribuí
        print('p_value = %.3f > alpha = %.2f' % (p_value, alpha))
        print('Não podemos rejeitar a hipótese nula, sugerindo que a amostra é proveniente de uma população normalmente distribuída')
        return True
    else:
        # Rejeitamos H_O, amostra não proveniente de uma população normalmente distribuída 
        print('p_value = %.3f <= alpha = %.2f' % (p_value, alpha))
        print('Podemos rejeitar a hipótese nula, sendo a amostra não proveniente de uma população normalmente distribuída')
        return False
