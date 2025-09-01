import requests
import pandas as pd
from bs4 import BeautifulSoup  # Note: you don't actually use BeautifulSoup here, but it's imported if you want later

HEADERS = {
    "User-Agent": "Divy Patel divypatel125710@gmail.com",
    "Accept-Encoding": "gzip, deflate",
    # "Host": "www.sec.gov"
}

def get_cik(ticker):
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print("Error fetching ticker data.")
        return None

    try:
        data = response.json()
    except Exception as e:
        print("Error parsing JSON:", e)
        return None

    for item in data.values():  # data is a dict with integer keys
        if item['ticker'].upper() == ticker.upper():
            return str(item['cik_str']).zfill(10)

    print(f"Ticker {ticker} not found.")
    return None

def get_recent_filings(cik, form_type="10-K", count=5):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        print(f"Failed to fetch filings for CIK {cik}")
        print("Status Code:", r.status_code)
        print("Response Text (truncated):", r.text[:500])
        return pd.DataFrame()

    data = r.json()
    filings = pd.DataFrame(data['filings']['recent'])
    filings = filings[filings['form'] == form_type]
    filings = filings[['form', 'filingDate', 'accessionNumber', 'primaryDocument']].head(count)
    filings['filing_url'] = filings.apply(
        lambda row: f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{row['accessionNumber'].replace('-', '')}/{row['primaryDocument']}",
        axis=1
    )
    return filings

if __name__ == "__main__":
    ticker = "AAPL"
    cik = get_cik(ticker)
    print("CIK:", cik)

    if cik:
        df_10k = get_recent_filings(cik, "10-K")
        df_10q = get_recent_filings(cik, "10-Q")
        print("\n10-K Filings:\n", df_10k)
        print("\n10-Q Filings:\n", df_10q)
    else:
        print("CIK not found.")
