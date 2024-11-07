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

#ARIMA
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import os

# Download datasets directly from Kaggle only if there are not already downloaded
if not os.path.exists("./dataset/DataSet1.csv"):
    os.system("kaggle datasets download -d mkechinov/ecommerce-purchase-history-from-electronics-store -p ./dataset --unzip")
if not os.path.exists("./dataset/DataSet2.csv"):
    os.system("kaggle datasets download -d datafiniti/electronic-products-prices -p ./dataset --unzip")
print("Datasets downloaded and extracted to './dataset' directory")


# paths to your datasets
ds1 = "./dataset/kz.csv"
ds2 = "./dataset/DatafinitiElectronicsProductsPricingData.csv"

# read the CSV files
dataset1 = pd.read_csv(ds1)
dataset2 = pd.read_csv(ds2)

# clean up duplicates
data1_cleaned = dataset1.drop_duplicates()
data2_cleaned = dataset2.drop_duplicates()

# colums - 'event_time', 'prices.amountMin', 'prices.amountMax'
time_series_data = data2_cleaned[['dateUpdated', 'prices.amountMin', 'prices.amountMax']]

# making sure 'event_time' is in datetime format
time_series_data['dateUpdated'] = pd.to_datetime(time_series_data['dateUpdated'])

# setting event_time as the index
time_series_data.set_index('dateUpdated', inplace=True)

# calculate the average price (min and max)
time_series_data['avg_price'] = (time_series_data['prices.amountMin'] + time_series_data['prices.amountMax']) / 2

# resample to weekly for smoother trends
weekly_prices = time_series_data['avg_price'].resample('W').mean()

# forward fill any missing values
weekly_prices_filled = weekly_prices.ffill()

# ADF test to check if data is stationary
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    if result[1] > 0.05:
        print("Series is not stationary")
    else:
        print("Series is stationary")

# run the test on weekly data
adf_test(weekly_prices_filled)

# differencing if needed
if adfuller(weekly_prices_filled)[1] > 0.05:
    stationary_data = weekly_prices_filled.diff().dropna()
else:
    stationary_data = weekly_prices_filled

# auto arima to pick best model
model = auto_arima(
    stationary_data,
    seasonal=False,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

# fit model
model.fit(stationary_data)

# forecast 4 weeks ahead
forecast_steps = 4
forecast, conf_int = model.predict(n_periods=forecast_steps, return_conf_int=True)

# create forecast dates
forecast_index = pd.date_range(stationary_data.index[-1], periods=forecast_steps+1, freq='W')[1:]
forecast_series = pd.Series(forecast, index=forecast_index)

# plot historical data and forecast
plt.figure(figsize=(10, 6))
plt.plot(weekly_prices_filled, label='historical avg prices')
plt.plot(forecast_series, label='forecasted prices', color='orange')
plt.fill_between(forecast_index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
plt.title('avg price forecast for next month')
plt.xlabel('date')
plt.ylabel('avg price')
plt.legend()
plt.show()
