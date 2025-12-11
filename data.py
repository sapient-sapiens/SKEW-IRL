import yfinance as yf 
import pandas_datareader.data as web 
import pandas as pd 
import datetime 

start_date = "1990-01-01"
end_date = datetime.date.today().strftime("%Y-%m-%d")

print("Fetching Market DAta (SKEW, VIX, SPX)") 
market_tickers = ["^SKEW", "^VIX", "^GSPC"]
market_data = yf.download(market_tickers, start=start_date, end=end_date)["Close"]
market_data.columns = ["SKEW", "VIX", "SPX"]
credit_data = web.DataReader("BAA10Y", "fred", start_date, end_date)
credit_data.columns = ["Credit_Spread"]
df = market_data.join(credit_data, how = "inner") 
df = df.ffill()
df = df.dropna() 
df["SPX_Return_1m"] = df["SPX"].pct_change(periods=21)

df["SKEW_Lag1"] = df["SKEW"].shift(1)

df = df.dropna()

print(f"Success! Final dataset has {len(df)} rows.")
print(df.head())

df.to_csv("skew_project_data.csv")
print("Saved to skew_project_data.csv")