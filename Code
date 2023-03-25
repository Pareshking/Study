import pandas as pd

import numpy as np

import yfinance as yf

import yahooquery as yq

import yahoofinancials as yfs

from datetime import datetime, timedelta

from tqdm import tqdm

# Define the start and end dates for the data

end_date = datetime.today().strftime('%Y-%m-%d')

start_date = (datetime.today() - timedelta(days=365*20)).strftime('%Y-%m-%d')

# Load symbol list

df = pd.read_csv('/content/ind_niftytotalmarket_list.csv')

# Create Yahoo symbol

df['Yahoo_Symbol'] = df.Symbol + '.NS'

symbol_list = df['Yahoo_Symbol'].tolist()

# Initialize empty dictionary for storing results

results_dict = {}

# Loop through each symbol and calculate the last 1, 2, 3, 6, 9, and 12 month returns, standard deviations, Sharpe Ratio, moving averages, highs, and market cap

for symbol in tqdm(symbol_list):

    # Get the industry for the symbol

    industry = df[df['Yahoo_Symbol'] == symbol]['Industry'].values[0]

    # Download adjusted close price data for the symbol

    data = yf.download(symbol, start=start_date, end=end_date, progress=False)['Adj Close']

    # Calculate the daily percentage change using ln(today's close/yesterday's close)

    daily_returns = np.log(data / data.shift(1))

    # Calculate the returns, standard deviations, and Sharpe ratio for each period

    periods = [21, 42, 63, 126, 189, 252]

    returns = [data.pct_change(periods=p).iat[-1] for p in periods if p != 'max']

    sds = [daily_returns.rolling(window=p).std().iat[-1] for p in [63, 126, 189, 252]]

    sharpe_ratios = [r / (sd * np.sqrt(p)) for r, sd, p in zip(returns[2:], sds, periods[2:])]

    

    # Calculate the moving averages and highs

    ma_periods = [20, 50, 100, 200]

    moving_averages = [data.rolling(window=p).mean().iat[-1] for p in ma_periods]

    high_52wk = data.rolling(window=252).max().iat[-1]

    ath = data.max()

    

    # Get the market cap for the symbol

    yq_ticker = yq.Ticker(symbol)

    market_cap = yq_ticker.summary_detail[symbol].get('marketCap')

    if market_cap:

        market_cap_formatted = '{:,.0f}'.format(market_cap / 10000000)

    else:

        market_cap_formatted = 'N/A'

    

    # Calculate the current price's percentage distance from 52-week high and all-time high

    last_close = data.iat[-1]

    high_52wk = data.rolling(window=252).max().iat[-1]

    if np.isnan(high_52wk):

        high_52wk = data.max()

    pct_from_52wk = (last_close - high_52wk) / high_52wk * 100

    pct_from_ath = (last_close - ath) / ath * 100

    # Remove .NS suffix

    symbol = symbol.replace(".NS", "")

    # Add the symbol, last close price, standard deviations, and returns to the results dictionary

    results_dict[symbol] = {

              'Symbol':symbol,

              'Industry': industry,

              'Price': '{:,.0f}'.format(last_close),

              'Market Cap': market_cap_formatted,

              '52W High': '{:,.0f}'.format(high_52wk),

              '52W Dist': '{:.2%}'.format(pct_from_52wk/100).replace('%', ''),

              'ATH': '{:,.0f}'.format(ath),

              'ATH Dist': '{:.2%}'.format(pct_from_ath/100).replace('%', ''),

              '20D MA': '{:,.0f}'.format(moving_averages[0]),

              '50D MA': '{:,.0f}'.format(moving_averages[1]),

              '100D MA': '{:,.0f}'.format(moving_averages[2]),

              '200D MA': '{:,.0f}'.format(moving_averages[3]),                            

              '1M ROC': '{:.0%}'.format(returns[0]),

              '2M ROC': '{:.0%}'.format(returns[1]),

              '3M ROC': '{:.0%}'.format(returns[2]),

              '3M SD': '{:.1%}'.format(sds[0]),

              '3M Sharpe': '{:.2f}'.format(sharpe_ratios[0]),

              '6M ROC': '{:.0%}'.format(returns[3]),

              '6M SD': '{:.1%}'.format(sds[1]),

              '6M Sharpe': '{:.2f}'.format(sharpe_ratios[1]),

              '9M ROC': '{:.0%}'.format(returns[4]),

              '9M SD': '{:.1%}'.format(sds[2]),

              '9M Sharpe': '{:.2f}'.format(sharpe_ratios[2]),

              '12M ROC': '{:.0%}'.format(returns[5]),

              '12M SD': '{:.1%}'.format(sds[3]),                          

              '12M Sharpe': '{:.2f}'.format(sharpe_ratios[3]),

    }

    

    # Convert results dictionary to a DataFrame

    df_results = pd.DataFrame.from_dict(results_dict, orient='index')

    # Save DataFrame to a CSV file

    df_results.to_csv('results.csv')

