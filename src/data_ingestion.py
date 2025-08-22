# src/data_ingestion.py

import yfinance as yf
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime

# -------------------------------
# 1. Structured Data (Yahoo Finance)
# -------------------------------
def get_stock_data(ticker="AAPL", period="6mo", interval="1d"):
    """
    Download stock price data from Yahoo Finance.
    Default: Apple (AAPL), last 6 months, daily interval.
    """
    stock = yf.download(ticker, period=period, interval=interval)
    stock.reset_index(inplace=True)
    return stock


# -------------------------------
# 2. Unstructured Data (Yahoo Finance RSS News)
# -------------------------------
def get_news_headlines(ticker="AAPL"):
    """
    Fetch latest news headlines for a given stock ticker from Yahoo Finance RSS.
    """
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    response = requests.get(url)

    headlines = []
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        for item in root.findall("./channel/item"):
            title = item.find("title").text
            pub_date = item.find("pubDate").text
            headlines.append({"date": pub_date, "headline": title})
    else:
        print("Failed to fetch news RSS feed")

    return pd.DataFrame(headlines)


# -------------------------------
# Main Runner
# -------------------------------
if __name__ == "__main__":
    # Example: Get Apple financial + news data
    stock_df = get_stock_data("AAPL")
    news_df = get_news_headlines("AAPL")

    print("Stock Data Sample:")
    print(stock_df.head())

    print("\nNews Headlines Sample:")
    print(news_df.head())

    # Save locally
    stock_df.to_csv("data/stock_data.csv", index=False)
    news_df.to_csv("data/news_data.csv", index=False)

    print("\n✅ Data saved inside /data folder!")
