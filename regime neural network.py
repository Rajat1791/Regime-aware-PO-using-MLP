import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "MA",
    "XOM", "CVX", "JNJ", "PG", "KO"
]

data = yf.download(
    tickers,
    start="2015-07-01",
    end="2025-07-01",
    progress=False
)["Close"]

returns = np.log(data / data.shift(1)).dropna()


window = 20

features = pd.DataFrame(index=returns.index)
features["market_vol"] = returns.std(axis=1).rolling(window).mean()
features["market_ret"] = returns.mean(axis=1).rolling(window).mean()

features = features.dropna()
returns = returns.loc[features.index]
vol_threshold = features["market_vol"].median()
features["regime"] = (features["market_vol"] > vol_threshold).astype(int)
# 0 = Calm, 1 = Volatile
X = torch.tensor(
    features[["market_vol", "market_ret"]].values,
    dtype=torch.float32
)
y = torch.tensor(features["regime"].values, dtype=torch.long)

class RegimeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.model(x)

model = RegimeNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(200):
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()

def equal_weight(n):
    return np.ones(n) / n
def defensive_weights(window_returns):
    vol = np.std(window_returns, axis=0)
    w = 1 / (vol + 1e-6)
    return w / np.sum(w)
portfolio_returns = []
strategy_labels = []

for i in range(60, len(features) - 21, 21):
    feat = torch.tensor(
        features.iloc[i][["market_vol", "market_ret"]].values,
        dtype=torch.float32
    )

    regime = torch.argmax(model(feat)).item()
    window_returns = returns.iloc[i-60:i].values

    if regime == 0:
        w = equal_weight(len(tickers))
        strategy_labels.append("Calm → EW")
    else:
        w = defensive_weights(window_returns)
        strategy_labels.append("Volatile → Defensive")

    realized = returns.iloc[i:i+21].values @ w
    portfolio_returns.extend(realized)
def run_equal_weight():
    w = equal_weight(len(tickers))
    return returns.values @ w

def run_defensive():
    out = []
    for i in range(60, len(returns)-21, 21):
        w = defensive_weights(returns.iloc[i-60:i].values)
        out.extend(returns.iloc[i:i+21].values @ w)
    return np.array(out)

ew_returns = run_equal_weight()
def_returns = run_defensive()
adaptive_returns = np.array(portfolio_returns)
def summarize(name, r):
    cum_ret = np.exp(np.sum(r))
    cvar_95 = np.mean(np.sort(-r)[-int(0.05*len(r)):])
    return [name, cum_ret, cvar_95]

results = pd.DataFrame([
    summarize("Equal Weight", ew_returns),
    summarize("Defensive Static", def_returns),
    summarize("Proposed Regime-Aware", adaptive_returns)
], columns=["Strategy", "Cumulative Return", "CVaR(95%)"])

print(results)
def worst_period_loss(returns, period=21):
    """
    returns: log returns array
    period: 21 = monthly, 5 = weekly
    """
    losses = []
    for i in range(0, len(returns)-period, period):
        losses.append(np.sum(returns[i:i+period]))
    return np.min(losses)

worst_ew = worst_period_loss(ew_returns, period=21)
worst_def = worst_period_loss(def_returns, period=21)
worst_prop = worst_period_loss(adaptive_returns, period=21)

print("Worst Monthly Loss (log-return):")
print("Equal Weight:", worst_ew)
print("Defensive Static:", worst_def)
print("Proposed Regime-Aware:", worst_prop)

Split returns by regime
regime_series = features["regime"].iloc[60:-21:21].values

ew_regime = []
prop_regime = []

idx = 0
for r in regime_series:
    ew_regime.append((r, ew_returns[idx:idx+21].sum()))
    prop_regime.append((r, adaptive_returns[idx:idx+21].sum()))
    idx += 21

ew_regime = pd.DataFrame(ew_regime, columns=["Regime", "Return"])
prop_regime = pd.DataFrame(prop_regime, columns=["Regime", "Return"])

print("Mean return by regime (0=Calm, 1=Volatile)")
print("Equal Weight:\n", ew_regime.groupby("Regime").mean())
print("Proposed:\n", prop_regime.groupby("Regime").mean())


Turnover analysis
def compute_turnover(weights_list):
    turnover = []
    for i in range(1, len(weights_list)):
        turnover.append(np.sum(np.abs(weights_list[i] - weights_list[i-1])))
    return np.mean(turnover)

# collect weights during switching
weights_history = []

for i in range(60, len(features)-21, 21):
    feat = torch.tensor(
        features.iloc[i][["market_vol","market_ret"]].values,
        dtype=torch.float32
    )
    regime = torch.argmax(model(feat)).item()
    window_returns = returns.iloc[i-60:i].values

    if regime == 0:
        w = equal_weight(len(tickers))
    else:
        w = defensive_weights(window_returns)

    weights_history.append(w)

turnover_prop = compute_turnover(weights_history)
print("Average Turnover (Proposed):", turnover_prop)

