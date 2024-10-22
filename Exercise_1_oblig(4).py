import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from scipy.optimize import minimize

# Download stock data
stocks = ['GJF.OL', 'ORK.OL', 'FRO.OL', 'WWI.OL', 'ABG.OL']
stock_names = {
    'GJF.OL': 'Gjensidige Forsikring ASA',
    'ORK.OL': 'Orkla ASA',
    'FRO.OL': 'Frontline Ltd',
    'WWI.OL': 'Wallenius Wilhelmsen ASA',
    'ABG.OL': 'ABG Sundal Collier'
}
#a)
daily_data = yf.download(stocks, period='1y', interval='1d')['Adj Close']
weekly_data = yf.download(stocks, period='2y', interval='1wk')['Adj Close']
# Calculate returns
daily_returns = np.log(daily_data / daily_data.shift(1)).dropna()
weekly_returns = np.log(weekly_data / weekly_data.shift(1)).dropna()
# Calculate expected returns and volatility
expected_daily_returns = daily_returns.mean() * 252
daily_volatility = daily_returns.std() * np.sqrt(252)
expected_weekly_returns = weekly_returns.mean() * 52
weekly_volatility = weekly_returns.std() * np.sqrt(52)

print("\nDaily expected Returns:")
print(expected_daily_returns)
print("\nDaily volatility:")
print(daily_volatility)
print("\nWeekly Expected Returns:")
print(expected_weekly_returns)
print("\nWeekly Volatility:")
print(weekly_volatility)

# Loop through each ticker
for ticker in stocks:
    plt.figure(figsize=(10, 6))

    # Plot histogram and density for daily returns
    plt.hist(daily_returns[ticker], bins=30, density=True, alpha=0.4, color='orange', label='Daily Histogram')
    sns.kdeplot(daily_returns[ticker], label='Daily Empirical Density', color='blue')

    mu_daily, std_daily = norm.fit(daily_returns[ticker])
    x_daily = np.linspace(daily_returns[ticker].min(), daily_returns[ticker].max(), 100)
    p_daily = norm.pdf(x_daily, mu_daily, std_daily)
    plt.plot(x_daily, p_daily, 'r--', linewidth=2, label='Daily Fitted Normal')

    # Plot histogram and density for weekly returns
    plt.hist(weekly_returns[ticker], bins=30, density=True, alpha=0.4, color='lime', label='Weekly Histogram')
    sns.kdeplot(weekly_returns[ticker], label='Weekly Empirical Density', color='slateblue')

    mu_weekly, std_weekly = norm.fit(weekly_returns[ticker])
    x_weekly = np.linspace(weekly_returns[ticker].min(), weekly_returns[ticker].max(), 100)
    p_weekly = norm.pdf(x_weekly, mu_weekly, std_weekly)
    plt.plot(x_weekly, p_weekly, 'k--', linewidth=2, label='Weekly Fitted Normal')


    plt.title(f"Daily vs Weekly Returns of {stock_names[ticker]}")
    plt.xlabel('Return')
    plt.ylabel('Density')
    plt.legend()

    plt.show()


#b)
# Calculate correlation and covariance matrices
corr_matrix_daily = daily_returns.corr()
cov_matrix_daily = daily_returns.cov() * 252
corr_matrix_weekly = weekly_returns.corr()
cov_matrix_weekly = weekly_returns.cov() * 52

print("\nDaily Correlation Matrix:")
print(corr_matrix_daily)
print("\nDaily Covariance Matrix:")
print(cov_matrix_daily)
print("\nWeekly Correlation Matrix:")
print(corr_matrix_weekly)
print("\nWeekly Covariance Matrix:")
print(cov_matrix_weekly)

#c)
# Function to calculate portfolio performance
def portfolio_return_volatility(weights, expected_weekly_returns, cov_matrix):
    portfolio_return = np.dot(weights, expected_weekly_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

# Function to calculate portfolio variance
def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

num_assets = len(stocks)

#Weights for stocks
initial_weights = num_assets * [1. / num_assets,]
#Make sure that the weights sum to 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
# Bounds for weights: weights between 0 and 1 (no short-selling)
bounds = tuple((0, 1) for asset in range(num_assets))

# Minimize variance in terms of the minimum-variance portfolio
opt_results = minimize(portfolio_variance, initial_weights, args=(cov_matrix_weekly.values,), method='SLSQP', bounds=bounds, constraints=constraints)
# Optimal weights for minimum-variance portfolio
min_var_weights = opt_results.x
# Expected return and volatility of the minimum-variance portfolio
min_var_return, min_var_volatility = portfolio_return_volatility(min_var_weights, expected_weekly_returns.values, cov_matrix_weekly.values)

print("\nMinimum Variance Portfolio:")
print("Weights:")
for i, stock in enumerate(stocks):
    print(f"{stock_names[stock]}: {min_var_weights[i]:.4f}")
print(f"Expected Return: {min_var_return:.4f}")
print(f"Volatility: {min_var_volatility:.4f}")

#Compute the efficient frontier
def efficient_frontier(expected_weekly_returns, cov_matrix_weekly, returns_range):
    efficient_portfolios = []
    num_assets = len(expected_weekly_returns)
    bounds = tuple((0, 1) for asset in range(num_assets))
    for target_return in returns_range:
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'eq', 'fun': lambda x: np.dot(x, expected_weekly_returns) - target_return})
        result = minimize(portfolio_variance, num_assets * [1. / num_assets], args=(cov_matrix_weekly,), method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            portfolio_volatility = np.sqrt(result.fun)
            efficient_portfolios.append({'Return': target_return, 'Volatility': portfolio_volatility, 'Weights': result.x})
        else:
            print("Optimization failed for target return:", target_return)
    return efficient_portfolios

# Define range of target returns
target_returns = np.linspace(expected_weekly_returns.min(), expected_weekly_returns.max(), 100)
# Compute efficient frontier
efficient_portfolios = efficient_frontier(expected_weekly_returns.values, cov_matrix_weekly.values, target_returns)

# Extract returns and volatilities
ef_returns = [p['Return'] for p in efficient_portfolios]
ef_volatilities = [p['Volatility'] for p in efficient_portfolios]
# Plot efficient frontier, I tried doing it with a for loop but it didn't work so I did it manually with a dataframe instead.
data = {
    'Ticker': ['ABG.OL', 'FRO.OL', 'GJF.OL', 'ORK.OL', 'WWI.OL'],
    'Company': ['ABG Sundal Collier', 'Frontline Ltd', 'Gjensidige Forsikring ASA', 'Orkla ASA', 'Wallenius Wilhelmsen ASA'],
    'Expected Return': [0.203643, 0.324811, 0.076677, 0.191109, 0.394796],
    'Volatility': [0.283856, 0.389489, 0.219141, 0.208403, 0.255246]
}

df = pd.DataFrame(data)
df.set_index('Ticker', inplace=True)
df.sort_values('Company', inplace=True)

#Plotting
plt.figure(figsize=(10, 7))
for index, row in df.iterrows():
    plt.scatter(row['Volatility'], row['Expected Return'], label=f"{row['Company']} ({index})")
plt.plot(ef_volatilities, ef_returns, 'b--', label='Efficient Frontier')
plt.scatter(min_var_volatility, min_var_return, c='green', marker='*', s=200, label='Minimum Variance Portfolio')

plt.title('Efficient Frontier')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.legend()
plt.grid(True)
plt.show()
# d)
# Removing Gjensidige Forsikring ASA from the portfolio
stocks_reduced = [stock for stock in stocks if stock != 'GJF.OL']
stock_names_reduced = {stock: stock_names[stock] for stock in stocks_reduced}

#Calculate cov_matrix again with reduced stocks
expected_weekly_returns_reduced = expected_weekly_returns[stocks_reduced]
cov_matrix_weekly_reduced = cov_matrix_weekly.loc[stocks_reduced, stocks_reduced]

num_assets = len(stocks_reduced)
initial_weights = num_assets * [1. / num_assets,]
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for asset in range(num_assets))

# Minimize variance to find the minimum-variance portfolio
opt_results = minimize(portfolio_variance, initial_weights, args=(cov_matrix_weekly_reduced.values,), method='SLSQP', bounds=bounds, constraints=constraints)
min_var_weights = opt_results.x

min_var_return, min_var_volatility = portfolio_return_volatility(
    min_var_weights,
    expected_weekly_returns_reduced.values,
    cov_matrix_weekly_reduced.values
)

print("\nMinimum Variance Portfolio without Gjensidige Forsikring ASA:")
print("Weights:")
for i, stock in enumerate(stocks_reduced):
    print(f"{stock_names_reduced[stock]}: {min_var_weights[i]:.4f}")
print(f"Expected Return: {min_var_return:.4f}")
print(f"Volatility: {min_var_volatility:.4f}")

#Compute the efficient frontier for the reduced portfolio
def efficient_frontier(expected_returns, cov_matrix, returns_range):
    efficient_portfolios = []
    num_assets = len(expected_returns)
    bounds = tuple((0, 1) for asset in range(num_assets))
    for target_return in returns_range:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return}
        )
        result = minimize(portfolio_variance, num_assets * [1. / num_assets], args=(cov_matrix,), method='SLSQP', bounds=bounds, constraints=constraints
        )
        if result.success:
            portfolio_volatility = np.sqrt(result.fun)
            efficient_portfolios.append({
                'Return': target_return,
                'Volatility': portfolio_volatility,
                'Weights': result.x
            })
        else:
            print("Optimization failed for target return:", target_return)
    return efficient_portfolios

target_returns_reduced = np.linspace(expected_weekly_returns_reduced.min(), expected_weekly_returns_reduced.max(), 100)

efficient_portfolios_reduced = efficient_frontier(
    expected_weekly_returns_reduced.values,
    cov_matrix_weekly_reduced.values,
    target_returns_reduced
)

ef_returns_reduced = [p['Return'] for p in efficient_portfolios_reduced]
ef_volatilities_reduced = [p['Volatility'] for p in efficient_portfolios_reduced]

data_reduced = {
    'Ticker': stocks_reduced,
    'Company': [stock_names_reduced[ticker] for ticker in stocks_reduced],
    'Expected Return': expected_weekly_returns_reduced.values,
    'Volatility': weekly_volatility[stocks_reduced].values
}

df_reduced = pd.DataFrame(data_reduced)
df_reduced.set_index('Ticker', inplace=True)
df_reduced.sort_values('Company', inplace=True)

plt.figure(figsize=(10, 7))

for index, row in df_reduced.iterrows():
    plt.scatter(row['Volatility'], row['Expected Return'], label=f"{row['Company']} ({index})")

# Plot the efficient frontier
plt.plot(ef_volatilities_reduced, ef_returns_reduced, 'b--', label='Efficient Frontier without GJF.OL')

plt.scatter(min_var_volatility, min_var_return, c='green', marker='*', s=200, label='Minimum Variance Portfolio without GJF.OL')

plt.title('Efficient Frontier after Removing Gjensidige Forsikring ASA')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.legend()
plt.grid(True)
plt.show()
