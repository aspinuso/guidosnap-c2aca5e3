import pandas as pd
import csv
import numpy as np
import plotly.express as px

class IndexDataMerging:
    def __init__(self, day, month, year, index):
        self.day = day
        self.month = month
        self.year = year
        self.index = index

    def Data_Frame_Day(self):
        YYYYMMDD=10000*self.year+100*self.month+self.day
        file_path='Guido-data/Data stations.txt'
        list=[210,215,235,240,249,260,269,270,275,277,279,280,283,310,340,344,350,370,375,377,380,391]
        list=[str(v) for v in list]
        df = pd.read_csv(file_path)
        df = df.loc[df['STN'].isin(list)]
        df = df[['STN', 'LOCATIE', 'POS_NB', 'POS_OL']]
        values=[]
        for station in list:
            index_path='Guido-data/Index files/FWI_'+station+'.txt'
            df_station=pd.read_csv(index_path)
            try:
                value=float(df_station.loc[df_station['date'] == YYYYMMDD, self.index].to_numpy()[0])
                values.append(value)
            except IndexError:
                values.append('NAN')        
        df[self.index]=values
        df=df[df[self.index] != 'NAN']
        df['shrt_info']=df[self.index].astype('int')
        return df
        
