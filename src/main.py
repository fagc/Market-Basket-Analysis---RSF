import kagglehub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Uncomment to download the dataset
# Download latest version
# path = kagglehub.dataset_download("mkechinov/ecommerce-purchase-history-from-electronics-store")
#path = kagglehub.dataset_download("datafiniti/electronic-products-prices")
# print("Path to dataset files:", path)

ds1 = "./dataset/DataSet1.csv"  
ds2 = "./dataset/DataSet2.csv"  
# Read the CSV file
df1 = pd.read_csv(ds1)
df2 = pd.read_csv(ds2)
print("df1 - ",df1.head())
print("df2-",df2.head())
# Know your data

#Shape of dataset
print('\nDS1 shape:', df1.shape)
print('\nDS2 Shape :', df2.shape)
#Dataset columns
print('\ncolumns in ds1 :', df1.keys())
print('\ncolumns in ds2 :', df2.keys())

# Remove duplicate rows
retail1 = df1.drop_duplicates(subset='order_id')
retail2 =df2.drop_duplicates(subset='id')

#revoe blank columns
retail2 = data_cleaned = df2.drop(columns=['Unnamed: 26', 'Unnamed: 27','Unnamed: 28', 'Unnamed: 29','Unnamed: 30'])  # Replace with actual column names

#filer data for demo purpose based on date

retail1['event_time'] = pd.to_datetime(retail1['event_time'])

retail1  = retail1[retail1['event_time']>='2020-06-30']

retail2['dateAdded'] = pd.to_datetime(retail2['dateAdded'])

retail2  = retail2[retail2['dateAdded'].dt.year >=2016]

print("filtered dataset1 head = ", retail1.head())
print("filtered dataset2 head = ", retail2.head())

print("Total rows after filtering dataset1", len(retail1))
print("Total rows after filtering dataset2",len(retail2))
#Unique values of categoriegs
print('\nUnique categories in ds1 :', len(retail1['category_code'].unique()))
print('\nUnique categories in ds2 :', len(retail2['categories'].unique()))

#dataset1
#Unique Customers
customers = retail1['user_id'].nunique()
print('\nUnique Customers in ds1 :', customers)

#Unique Items
stock = retail1['product_id'].nunique()
print('\nUnique items sold in ds1 :',stock)

#Unique brands
brand = retail1['brand'].nunique()
print('\nUnique brand sold in ds1 :',brand)

#dataset2

#Unique Items
stock2 = retail2['name'].nunique()
print('\nUnique items sold in ds2:',stock2)
#Unique Items
brand2 = retail2['brand'].nunique()
print('\nUnique brands sold in ds2 :',brand2)


#EDA
#DS1
print('\n-------------DataSet1----------')
#Checking the data type
print(retail1.dtypes)
#Describe dataset
print(retail1.describe())
#Removing Null values of CustomerID
retail1 = retail1[retail1['user_id'].notna()]
#Checking Null Values
retail1.isnull().sum()

#DS2
print('\n-------------DataSet2----------')
#Checking the data type
print(retail2.dtypes)
#Describe dataset
print(retail2.describe())
#Removing Null values of CustomerID
retail2 = retail2[retail2['id'].notna()]
#Checking Null Values
retail2.isnull().sum()