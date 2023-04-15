
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

#Get the stock data from yfinance
#stock_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'BABA', 'BRK-B', 'TCEHY', 'JPM', 'V'] #2019Q4
stock_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'TSM', 'TCEHY']#2021Q4

start_date = '2021-01-01'
mid_date = '2021-12-31'
end_date = '2022-12-31'
#stock_data = yf.download(stock_tickers, start=start_date, end=end_date)['Adj Close']
stock_data = pd.read_csv('stock_data2021-2022.csv', index_col=0)
#%%

ind_returns = pd.DataFrame(np.log(stock_data.loc[start_date: mid_date] / stock_data.loc[start_date: mid_date].shift(1)))
#ind_returns = (ind_returns - ind_returns.mean()) / ind_returns.std()

ind_returns.plot(figsize=(15,12), subplots=True, layout=(4,3))
ind_returns.hist(figsize=(15,12), bins=50)

#Calaulte returns/risk 
ret_std = ind_returns.mean()*252 / (ind_returns.std()*(252**0.5))

#%%
#Calculate the returns and covariance matrix
returns = ind_returns.mean() * 252
cov_matrix = ind_returns.cov()*252

print('returns: ', returns)
print('covariance matrix: ', cov_matrix)


#%%
def portfolio_objective(weights, returns, cov_matrix, risk_aversion=0.5):
    portfolio_return = np.sum(weights * returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    #objective = portfolio_return - risk_aversion * portfolio_variance
    objective = portfolio_return / (portfolio_variance**0.5)
    return -objective

def maximize_mean_variance_utility(returns, cov_matrix, risk_aversion=0.5):
    num_assets = returns.shape[0]
    initial_weights = np.ones(num_assets) / num_assets
    constraints = ({'type': 'eq', 'fun': lambda x: 1-np.sum(x)})
    bounds = [(0, 1) for i in range(num_assets)]
    result = minimize(portfolio_objective, initial_weights, args=(returns, cov_matrix, risk_aversion),
                     method='SLSQP', constraints=constraints, bounds=bounds)
    #min method: 'L-BFGS-B' 'trust-constr' 'SLSQP' 'Newton-CG’
    return result.x

risk_aversion = 0.5
MVO_weights = maximize_mean_variance_utility(returns, cov_matrix, risk_aversion)


#%%
# Get the price data of SPY and the portfolio

benchmark_data = yf.download('SPY', start=mid_date, end=end_date)['Adj Close']
#portfolio_data = (stock_data[stock_tickers].loc[mid_date:end_date] @ MVO_weights).to_frame()
portfolio_data = (stock_data.loc[mid_date:end_date] @ MVO_weights).to_frame()
ew_portfolio = (stock_data.loc[mid_date:end_date] @ (np.ones(len(stock_tickers))/len(stock_tickers))).to_frame()

#%%
# Calculate the daily returns
benchmark_returns = pd.DataFrame(np.log(benchmark_data / benchmark_data.shift(1)))
portfolio_returns = pd.DataFrame(np.log(portfolio_data / portfolio_data.shift(1)))
ew_returns = pd.DataFrame(np.log(ew_portfolio / ew_portfolio.shift(1)))

#calculate sharpe ratio
benchmark_sharpe = benchmark_returns.mean()*252 / (benchmark_returns.std()*(252**0.5))
portfolio_sharpe = portfolio_returns.mean()*252 / (portfolio_returns.std()*(252**0.5))
ew_sharpe = ew_returns.mean()*252 / (ew_returns.std()*(252**0.5))
  
#calculate cumulative returns                   
cum_benchmark = (benchmark_returns+1).cumprod()
cum_benchmark.columns=['SPY']
cum_portfolio = (portfolio_returns+1).cumprod()
cum_portfolio.columns=['MVO portfolio']
cum_ew = (ew_returns+1).cumprod()
cum_ew.columns=['Equally weighted']

df = pd.concat([cum_benchmark, cum_portfolio, cum_ew], axis=1)
# Plot the cumulative returns
cum_return_plot = df.plot(figsize=(15,12), title='Cumulative Returns: MVO portfolio v.s. SPY v.s. Equally weighted portfolio\n'+mid_date+' to '+end_date)
cum_return_plot.set_ylabel('Cumulative Returns')
#plt.plot(benchmark_returns, label='Nasdaq 100')
#plt.plot(portfolio_returns, label='MVO Portfolio')
#plt.show()
