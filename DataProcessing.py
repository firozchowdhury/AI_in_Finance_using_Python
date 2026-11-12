#loading csv file using pandas
import pandas as pd
df = pd.read_csv('Financials4.csv')
print(df.head()) #see the first 5 rows of the dataframe

print(df.shape) #see the number of rows and columns in the dataframe
print(df.info()) #see the data types of each column and if there are any missing values
print(df.columns) #see the column names in the dataframe
print(df.isna().sum()) #see the number of missing values in each column

