import kagglehub
import pandas as pd
#Uncomment to download the dataset
# Download latest version
# path = kagglehub.dataset_download("mkechinov/ecommerce-purchase-history-from-electronics-store")
#path = kagglehub.dataset_download("datafiniti/electronic-products-prices")
# print("Path to dataset files:", path)

ds1 = "./dataset/DataSet1.csv"  
ds2 = "./dataset/DataSet2.csv"  
# Read the CSV file
dataset1 = pd.read_csv(ds1)
dataset2 = pd.read_csv(ds2)

#data cleanup

# Remove duplicate rows
data1_cleaned = dataset1.drop_duplicates()
data2_cleaned = dataset2.drop_duplicates()

