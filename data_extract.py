import yfinance as yf
import os

# Download associated stock data into csv files.
stocklist = ["JNJ", "MSFT", "TSM", "WMT", "PG", "NVDA", "DIS", "MMM", "AMD"]
start_date = "2012-01-01"
end_date = "2018-12-31"
change_dates = False
data = {}
for stock in stocklist:
    resulting_filename = "data_"+stock+".csv"
    if (os.path.exists(resulting_filename) and not change_dates):
        print(resulting_filename, "exists")
        continue
    if change_dates:
        print(resulting_filename+": changing dates... retrieving data...")
    else:
        print(resulting_filename, "does not exist... retrieving data...")
    ticker = yf.Ticker(stock)
    data[stock] = ticker.history(start = start_date, end = end_date)
    print(data[stock].head(20)) # prints first 20 lines of downloaded dataframe
    data[stock].to_csv(path_or_buf=resulting_filename)