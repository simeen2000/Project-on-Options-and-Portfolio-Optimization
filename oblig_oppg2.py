import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import yfinance as yf
import pandas as pd


r = 0.045
S0 = 1000

sigma_n = [0.15, 0.30, 0.45]
Time = [1/12, 3/12, 4/12]

#Make an array of K_1, K_2, K_3 
option_price = [0.7, 0.9, 1, 1.1, 1.3]
K_n = [S0 * n for n in option_price]

#a)
#Function for calculating call price using Black-Scholes 
def black_scholes(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

call_prices_list = []

#plot
fig, axs = plt.subplots(1, len(Time), figsize=(18, 6))
for index, T in enumerate(Time):
    ax = axs[index]
    for sigma in sigma_n:
        call_prices = []
        for K in K_n:
            call_price = black_scholes(S0, K, T, r, sigma)
            call_prices.append(call_price)
            call_prices_list.append({'Time': T * 12, 'Sigma': sigma * 100, 'Strike Price': K, 'Call Option Price': call_price
            })
        ax.plot(K_n, call_prices, marker='o', label=f'σ = {sigma*100:.0f}%')
    ax.set_title(f'Call Option Prices for T = {T*12:.0f} Months')
    ax.set_xlabel('Strike Price (K)')
    ax.set_ylabel('Call Option Price (C)')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# Create a DataFrame from the call_prices_list
call_prices_df = pd.DataFrame(call_prices_list)
#Organize data
df = call_prices_df.pivot_table(
    index=['Time', 'Strike Price'],
    columns='Sigma',
    values='Call Option Price'
)

pd.options.display.float_format = '{:.2f}'.format
print(df)
#b)
#Function for calculating implied volatility using the hint
def find_implied_volatility(S0, K, T, r, market_price, tol=0.005, max_iter=100):
    # Objective function for implied volatility
    def objective(sigma):
        return black_scholes(S0, K, T, r, sigma) - market_price
    sigma_lower = 1e-6
    sigma_upper = 2.0

    f_lower = objective(sigma_lower)
    f_upper = objective(sigma_upper)

    if f_lower * f_upper > 0:
        # No sign change: adjust bounds or return NaN
        return np.nan

    for _ in range(max_iter):
        sigma_mid = (sigma_lower + sigma_upper) / 2
        f_mid = objective(sigma_mid)

        if (sigma_upper - sigma_lower) / 2 < tol:
            return sigma_mid

        if f_mid * f_lower < 0:
            sigma_upper = sigma_mid
            f_upper = f_mid
        else:
            sigma_lower = sigma_mid
            f_lower = f_mid

    return (sigma_lower + sigma_upper) / 2

#Using NVDA
ticker = yf.Ticker('NVDA')
current_price = ticker.history(period="1d")['Close'].iloc[-1]
S0 = current_price  #Underlying asset price
#Expiration dates for NVDA options
expirations = ticker.options
#Colect option with expiration 2025-04-17
expiration = expirations[11]
option_chain = ticker.option_chain(expiration)

#Extract call options
calls = option_chain.calls

#Time to expiration
today = pd.Timestamp('today').normalize()
expiration_date = pd.to_datetime(expiration)
T = (expiration_date - today).days / 365.0

strike_prices = []
implied_volatilities = []

for idx, row in calls.iterrows():
    strike = row['strike']
    market_price = row['lastPrice']

    if np.isnan(market_price) or market_price == 0:
        continue

    iv = find_implied_volatility(S0, strike, T, r, market_price)
    if np.isnan(iv):
        continue  
    strike_prices.append(strike)
    implied_volatilities.append(iv)

# Plot Implied Volatility vs Strike Prices
plt.figure(figsize=(10, 6))
plt.plot(strike_prices, implied_volatilities, marker='o', linestyle='-', color='blue')
plt.title(f'Implied Volatility vs Strike Prices for NVDA (Expiration: {expiration})')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.grid(True)
plt.show()
print(expirations)