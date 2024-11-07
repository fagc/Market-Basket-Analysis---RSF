import kagglehub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

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
retail1 = retail1.dropna()

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
retail2 = retail2.dropna()

#Boxplot for UnitPrice
sns.boxplot(x=retail1['price'])


# Plotting a Histogram
retail1.user_id.value_counts().nlargest(20).plot(kind='bar', figsize=(10,5))
plt.title('Top 20 Customers in terms of Purchase')
plt.ylabel('Number of Purchases')
plt.xlabel('Customer ID')
plt.show()

# Plotting a Histogram
retail1.category_code.value_counts().nlargest(20).plot(kind='bar', figsize=(10,5))
plt.title('Purchase count of top 20 Items')
plt.ylabel('Number of Purchases')
plt.xlabel('Items')
plt.show()

# Plotting a Histogram
retail1.brand.value_counts().nlargest(20).plot(kind='bar', figsize=(10,5))
plt.title('Top 20 Brands in terms of Purchase')
plt.ylabel('Number of Purchases')
plt.xlabel('Brand')
plt.show()

#apriori
#clean data for apriori
retail1 = retail1.astype(bool).astype(int)  # Convert boolean to binary if needed
retail1 = retail1.apply(pd.to_numeric, errors='coerce').fillna(0)  # Ensure numeric format

# Run Apriori

# use the apriori method to find frequent patterns
min_support = 0.002

freq_itemsets = apriori(retail1, min_support=0.02, use_colnames=True)

## we can add the length of the itemsets as a column
freq_itemsets = (
    freq_itemsets
    .assign(
        length=lambda df_: df_.itemsets.apply(len),  # apply the len function
    )
)
print(freq_itemsets)

# Function to get top 5 itemsets by length
def top_itemsets_by_length(df, length, top_n=5):
    return df[df['length'] == length].nlargest(top_n, 'support')

# Cal the funtion for length 1, 2, 3
top_length_1 = top_itemsets_by_length(freq_itemsets, length=1)
top_length_2 = top_itemsets_by_length(freq_itemsets, length=2)
top_length_3 = top_itemsets_by_length(freq_itemsets, length=3)

print("Top 5 Itemsets Length 1:")
print(top_length_1[['itemsets', 'support']])
print("\nTop 5 Itemsets Length 2:")
print(top_length_2[['itemsets', 'support']])
print("\nTop 5 Itemsets Length 3:")
print(top_length_3[['itemsets', 'support']])