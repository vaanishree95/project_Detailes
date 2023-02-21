import mysql.connector

mydb = mysql.connector.connect(
                    host='localhost',
                    user='root',
                    password='', 
                    port=3307,
                    database='product sales',
                )



###### read dabase from mysql to dataframe ############
#import pymysql
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from feature_engine.outliers import Winsorizer
import matplotlib as mpl
import matplotlib.pyplot as plt
import autots 
from autots import AutoTS
import pickle
from statsmodels.tsa.seasonal import seasonal_decompose

pharma_df  = pd.read_sql_query('SELECT * FROM salesdaily', mydb) ##fetch the complete table                             
pharma_df1  = pd.read_sql_query('SELECT * FROM salesdaily', mydb) ##fetch the complete table                             


def preprocess():

    duplicate = pharma_df.duplicated()
    duplicate
    sum(duplicate)
    ##pharma_df = pharma_df.drop_duplicates()
    
#### no duplicates
   

    pharma_df.isna().sum()
    pharma_df.dropna(axis=1, inplace=True)
    

     
      
 
#######################################################################



    for col in pharma_df.columns:
        print("winsoring the ", col)
        if (((pharma_df[col].dtype)=='float64') | ((pharma_df[col].dtype)=='int64')):
                winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                                          tail='both', # cap left, right or both tails 
                                          fold=1.5,
                                          variables=[col])
                pharma_df[col] = winsor.fit_transform(pharma_df[[col]])
        else:
                pharma_df[col] = pharma_df[col]
            
            
    x = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']
    for i in x:
        result = seasonal_decompose(pharma_df[x].rolling(7, center=True).mean().dropna(), freq=7,model = "additive",filt=None)
        return pharma_df

            
############## Model building ###################


#train = pharma_df.head(1741) 
#test = pharma_df.tail(365)

def model_def(ab):
    model_AB = AutoTS(
                forecast_length=7,
                frequency='infer',
                prediction_interval=0.95,
                ensemble=None,
                model_list="fast", 
                transformer_list="fast",  
                drop_most_recent=1,
                max_generations=4,
                num_validations=2,
                validation_method="backwards")
        
     
  
     
    model_AB = model_AB.fit(
                pharma_df,
                date_col="datum",
                value_col= "M01AB"
                )  

 
    ## saving model to disk
    pickle.dump(model_AB, open('model_AB.pkl', 'wb'))
    
    ## loading model to compare the results
    model_AB = pickle.load(open('model_AB.pkl', 'rb'))
    

    
###################################################################   


    model_AE = AutoTS(
                forecast_length=7,
                frequency='infer',
                prediction_interval=0.95,
                ensemble=None,
                model_list="fast", 
                transformer_list="fast",  
                drop_most_recent=1,
                max_generations=1,
                num_validations=1,
                validation_method="backwards")
    model_AE = model_AE.fit(
                pharma_df,
                date_col="datum",
                value_col= "M01AE"
                )  

     
    ## saving model to disk

    pickle.dump(model_AE, open('model_AE.pkl', 'wb'))
    
    
    model_AE = pickle.load(open('model_AE.pkl', 'rb'))
    
 ##################################################################   


    model_N02BA = AutoTS(
                 forecast_length=7,
                 frequency='infer',
                 prediction_interval=0.95,
                 ensemble=None,
                 model_list="fast", 
                 transformer_list="fast",  
                 drop_most_recent=1,
                 max_generations=1,
                 num_validations=1,
                 validation_method="backwards")
    model_N02BA = model_N02BA.fit(
                 pharma_df,
                 date_col="datum",
                 value_col= "N02BA"
                 )  

      
     ## saving model to disk

    pickle.dump(model_N02BA, open('model_N02BA.pkl', 'wb'))
     
     
    model_N02BA = pickle.load(open('model_N02BA.pkl', 'rb'))
    
    
 ######################################################################   
    
    model_N02BE = AutoTS(
                 forecast_length=7,
                 frequency='infer',
                 prediction_interval=0.95,
                 ensemble=None,
                 model_list="fast", 
                 transformer_list="fast",  
                 drop_most_recent=1,
                 max_generations=1,
                 num_validations=1,
                 validation_method="backwards")
    model_N02BE = model_N02BE.fit(
                 pharma_df,
                 date_col="datum",
                 value_col= "N02BE"
                 )  

      
     ## saving model to disk

    pickle.dump(model_N02BE, open('model_N02BE.pkl', 'wb'))
     
     
    model_N02BE = pickle.load(open('model_N02BE.pkl', 'rb'))
    
    
 ######################################################################

    model_N05B = AutoTS(
                 forecast_length=7,
                 frequency='infer',
                 prediction_interval=0.95,
                 ensemble=None,
                 model_list="fast", 
                 transformer_list="fast",  
                 drop_most_recent=1,
                 max_generations=1,
                 num_validations=1,
                 validation_method="backwards")
    model_N05B = model_N05B.fit(
                 pharma_df,
                 date_col="datum",
                 value_col= "N05B"
                 )  

      
     ## saving model to disk

    pickle.dump(model_N05B, open('model_N05B.pkl', 'wb'))
     
     
    model_N05B = pickle.load(open('model_N05B.pkl', 'rb'))
    
    
    
 #####################################################################

    model_N05C = AutoTS(
                 forecast_length=7,
                 frequency='infer',
                 prediction_interval=0.95,
                 ensemble=None,
                 model_list="fast", 
                 transformer_list="fast",  
                 drop_most_recent=1,
                 max_generations=1,
                 num_validations=1,
                 validation_method="backwards")
    model_N05C = model_N05C.fit(
                 pharma_df,
                 date_col="datum",
                 value_col= "N05C"
                 )  

      
     ## saving model to disk

    pickle.dump(model_N05C, open('model_N05C.pkl', 'wb'))
     
     
    model_N05C = pickle.load(open('model_N05C.pkl', 'rb'))  
       
#######################################################################
    model_R03 = AutoTS(
                  forecast_length=7,
                  frequency='infer',
                  prediction_interval=0.95,
                  ensemble=None,
                  model_list="fast", 
                  transformer_list="fast",  
                  drop_most_recent=1,
                  max_generations=1,
                  num_validations=1,
                  validation_method="backwards")
    model_R03 = model_R03.fit(
                  pharma_df,
                  date_col="datum",
                  value_col= "R03"
                  )  

       
      ## saving model to disk

    pickle.dump(model_R03, open('model_R03.pkl', 'wb'))
      
      
    model_R03 = pickle.load(open('model_R03.pkl', 'rb'))  
           
    


##########################################################################

    model_R06 = AutoTS(
                  forecast_length=7,
                  frequency='infer',
                  prediction_interval=0.95,
                  ensemble=None,
                  model_list="fast", 
                  transformer_list="fast",  
                  drop_most_recent=1,
                  max_generations=1,
                  num_validations=1,
                  validation_method="backwards")
    model_R06 = model_R06.fit(
                  pharma_df,
                  date_col="datum",
                  value_col= "R06"
                  )  

       
      ## saving model to disk

    pickle.dump(model_R06, open('model_R06.pkl', 'wb'))
      
      
    model_R06 = pickle.load(open('model_R06.pkl', 'rb'))  
           

    return model_AB, model_AE, model_N02BA, model_N02BE, model_N05B, model_N05C, model_R03, model_R06

pipe = Pipeline([('preprocess', preprocess(), 
                  'model_build', model_def("M01AB"), 
                  'model_build', model_def("M01AE"),
                  'model_build', model_def("N02BA"),
                  'model_build', model_def("N02BE"),
                  'model_build', model_def("N05B"),
                  'model_build', model_def("N05C"),
                  'model_build', model_def("R03"),
                  'model_build', model_def("R06"))])     
    