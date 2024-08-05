# Portfolio Optimization

This repository contains a Python implementation of portfolio optimization using historical stock data. The project includes:

- Data collection from Yahoo Finance.
- Calculation of daily returns and annualized statistics.
- Optimization of asset weights to achieve maximum Sharpe ratio and minimum volatility.
- Visualization of the efficient frontier and simulated portfolios.

Key features:
- Downloads historical stock data for multiple tickers.
- Computes daily returns, mean returns, and covariance matrix.
- Utilizes quadratic programming (cvxopt) to find optimal asset allocations.
- Generates and visualizes the efficient frontier with random portfolios.

## Setup
pip install yfinance pandas numpy cvxopt matplotlib
