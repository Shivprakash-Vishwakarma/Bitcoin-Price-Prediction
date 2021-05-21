import csv
from Historic_Crypto import LiveCryptoData
from Historic_Crypto import HistoricalData

new = HistoricalData('BTC-USD', 86400, '2020-06-01-00-00').retrieve_data()
print(new)

new.to_csv("bitt.csv")
print("created")
