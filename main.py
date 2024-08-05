import yfinance as yf
import pandas as pd
import os
import numpy as np
import cvxopt as opt
from cvxopt import solvers
import matplotlib.pyplot as plt

########
# DATA #
########

# Tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

# Timeframe
start = '2010-01-01'
end = '2020-01-01'

# Download and filtering
data = yf.download(tickers, start=start, end=end)['Adj Close']
os.makedirs('data', exist_ok=True)
data.to_csv('data/stock_prices.csv')
data = pd.read_csv('data/stock_prices.csv', index_col='Date', parse_dates=True)
data = data.dropna()
returns = data.pct_change().dropna()  # Calculate daily returns
returns.to_csv('data/daily_returns.csv')

################
# Optimization #
################

returns = pd.read_csv('data/daily_returns.csv', index_col='Date', parse_dates=True)

# Annualize mean returns and covariance matrix
mean_returns = returns.mean() * 252  # Annualized mean returns
cov_matrix = returns.cov() * 252  # Annualized covariance matrix
n = len(mean_returns)  # Number of assets

# cvxopt matrices (quadratic programming setup)
P = opt.matrix(cov_matrix.values)  # Covariance matrix
q = opt.matrix(np.zeros(n))  # Zero vector
G = opt.matrix(np.diag(np.ones(n) * -1))  # Constraint for non-negative weights (Gw <= h)
h = opt.matrix(np.zeros(n))  # Zero vector (Gw <= h)  <=> (w >= 0)
A = opt.matrix(np.ones(n), (1, n))  # Constraint for weights summing to 1
b = opt.matrix(1.0)  # Scalar (Aw = b)

# Solve the optimization problem
solvers.options['show_progress'] = False
solution = solvers.qp(P, q, G, h, A, b)
weights = np.array(solution['x']).flatten()

# Save optimal weights
optimal_weights = pd.Series(weights, index=tickers, name='weights')
optimal_weights.to_csv('data/optimal_weights.csv')

print("Optimal Weights:")
print(optimal_weights)


######################
# Efficient Frontier #
######################

# Generate random portfolios, calculate their returns, volatilities, and Sharpe ratios
def generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate=0.0):
    results = np.zeros((3, num_portfolios))  # Columns for return, volatility, Sharpe ratio
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)  # Normalize weights
        weights_record.append(weights)

        portfolio_return = np.sum(weights * mean_returns)  
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev

        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = sharpe_ratio

    return results, weights_record

# Display efficient frontier with simulated portfolios showcasing the MVP and the Maximum Sharpe Ratio portfolio
def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights = generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[1, max_sharpe_idx], results[0, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=tickers, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100, 2) for i in max_sharpe_allocation.allocation]

    min_vol_idx = np.argmin(results[1])
    sdp_min, rp_min = results[1, min_vol_idx], results[0, min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index=tickers, columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100, 2) for i in min_vol_allocation.allocation]

    print("-"*80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualized Return:", round(rp, 2))
    print("Annualized Volatility:", round(sdp, 2))
    print("\n")
    print(max_sharpe_allocation)
    print("-"*80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualized Return:", round(rp_min, 2))
    print("Annualized Volatility:", round(sdp_min, 2))
    print("\n")
    print(min_vol_allocation)

    plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='YlGnBu', marker='o')
    plt.scatter(sdp, rp, marker='*', color='r', s=500, label='Maximum Sharpe Ratio')
    plt.scatter(sdp_min, rp_min, marker='*', color='g', s=500, label='Minimum Volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Returns')
    plt.colorbar(label='Sharpe Ratio')
    plt.legend(labelspacing=0.8)
    plt.show()


# Display efficient frontier with random portfolios
num_portfolios = 10000
risk_free_rate = 0.0378  # 10 year US treasury rate August 05 2024
print('----------------------------------')
print('----------------------------------')
print('Timeframe: From', start, 'to', end)
display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)

