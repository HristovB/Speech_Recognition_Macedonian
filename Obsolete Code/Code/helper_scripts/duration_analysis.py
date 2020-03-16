# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 21:08:46 2019

@author: Marija
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json('d:/DIPLOMSKA/train_corpus_allSubjects.json', lines=True)

x = df['duration']

b = [ x for x in range(0,50,5)]

plt.hist(x,bins=b, rwidth=0.95, color='green')
plt.xticks(b)
plt.xlabel('Времетраење (секунди)')
plt.ylabel('Број на примероци (аудио записи)')
plt.title('')
plt.savefig('dataset_histogram.png',dpi=600)

df_new = pd.DataFrame()

for index, row in df.iterrows():
    row['key']=row['key'].split('\\')[4]
    #print(row['key'].split('\\')[4])
    df_new = df_new.append(row)

sub1 = df_new.loc[(df_new['key']=='1') | (df_new['key']=='3')]
sub1 = pd.DataFrame()
sub1 = df_new.loc[((df_new['key']=='1') | (df_new['key']=='3') | (df_new['key']=='6') | (df_new['key']=='9') | (df_new['key']=='12')]

for index,row in sub1.iterrows():
    if row['duration'] > 20:
        sub1.drop(index, inplace=True)
		
print(np.mean(sub1['duration']))
print(np.std(sub1['duration']))