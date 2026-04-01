import pandas as pd
from astmhSupportFunctions_6april2024 import populateKeywordFeatureVector_fn

kws_file = open('/workspace/code/data/keywords_8april2024.txt', 'r')
kws = kws_file.read()

kws_list = kws.split(',')
kws_list[-1] = kws_list[-1][:-1]

train_data_path = '/workspace/code/data/train_split_merged_1.xlsx'
df = pd.read_excel(train_data_path)

df['kw_vector'] = df['abstractText'].apply(lambda x: populateKeywordFeatureVector_fn(x,
                                                                                     kws_list,
                                                                                     weights=[0, 1]))
print(df.iloc[50]['kw_vector'])
