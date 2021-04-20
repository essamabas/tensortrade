
# %% [markdown]
# # Include Libraries

# %%
# setup dependencies
import inspect
import sys
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
rootdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(rootdir)
parentdir


# %%
import pandas as pd
import tensortrade.env.default as default

from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
# Make a stream of closing prices to make orders on
from tensortrade.oms.instruments import USD, Instrument, Quantity
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.agents import DQNAgent
from tensortrade.env.default.renderers import PlotlyTradingChart, FileLogger, MatplotlibTradingChart


import gym
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines import DQN, PPO2, A2C
from stable_baselines.gail import generate_expert_traj

import ta
import numpy as np
import datetime
from scipy.signal import argrelextrema
import numpy as np
import yfinance as yf

# silence warnings
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')

# Use these commands - to reload sources, while development
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %% [markdown]
# # Helper Functions

# %%
def download_data(symbol: str, 
                  start_date: str, 
                  end_date: str = datetime.date.today().strftime('%Y-%m-%d'),
                  plot: bool = False) -> pd.DataFrame:
    # download Data
    df = yf.download(symbol, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df.columns = [name.lower() for name in df.columns]
    df.drop(columns=["adj close","volume"],inplace=True)
    df.set_index("date",inplace=True)
    if plot:
      df['close'].plot()

    return df





def __classify(self, current_index,df_min,df_max):
    '''
    Apply Local/Min - Max analysis
    '''
    if current_index in df_min.index:
        return 1 # buy-decision
    elif current_index in df_max.index:
        return -1 # sell-decision
    else:  # otherwise... it's a 0!
        return 0  # hold-decision

def find_loc_min_max(data: pd.DataFrame, 
                     order_of_points=7,
                     symbol: str = "symbol",
                     plot:bool = False):
    '''
      Find local peaks
    '''
    df_min_ts = data.iloc[argrelextrema(data.org_close.values, np.less_equal, order=order_of_points)[0]].astype(np.float32)
    df_max_ts = data.iloc[argrelextrema(data.org_close.values, np.greater_equal, order=order_of_points)[0]].astype(np.float32)

    df_min_ts = df_min_ts.iloc[:, 0:5]
    df_max_ts = df_max_ts.iloc[:, 0:5]

    if plot:
      import plotly.graph_objects as go
      fig = go.Figure(data= go.Scatter(
          x=data.index,
          y=data['org_close'],
          name = symbol
      ))
      #fig = go.Figure([go.Scatter(x=df['Date'], y=df['AAPL.High'])])                
      fig.add_trace(go.Scatter(mode="markers", x=df_min_ts.index, y=df_min_ts['org_close'], name="min",marker_color='rgba(0, 255, 0, .9)'))
      fig.add_trace(go.Scatter(mode="markers", x=df_max_ts.index, y=df_max_ts['org_close'], name="max",marker_color='rgba(255, 0, 0, .9)'))

      config = {'displayModeBar': False}
      fig.show(config=config)


    return df_min_ts, df_max_ts

def create_trade_env(quotes, observations ,symbol):

  # Add features
  features = []
  #exclude "date/Column [0]" from observation - start from column 1
  for c in observations.columns[0:]:
      s = Stream.source(list(observations[c]), dtype="float").rename(observations[c].name)
      features += [s]
  feed = DataFeed(features)
  feed.compile()

  # define exchange - needs to specify Price-Quote Stream
  exchange  = Exchange("sim-exchange", service=execute_order)(
      Stream.source(list(quotes["close"]), dtype="float").rename(str("USD-{}").format(symbol))
  )

  # add current cash, initial-asset
  cash = Wallet(exchange, 10000 * USD)
  asset = Wallet(exchange, 0 * Instrument(symbol, 2, symbol))

  # initialize portfolio - base currency USD
  portfolio = Portfolio(
      base_instrument = USD, 
      wallets = [
          cash,
          asset
      ]
  )

  # add element for rendered feed
  renderer_feed = DataFeed([
      Stream.source(list(observations.index)).rename("date"),
      Stream.source(list(observations["open"]), dtype="float").rename("open"),
      Stream.source(list(observations["high"]), dtype="float").rename("high"),
      Stream.source(list(observations["low"]), dtype="float").rename("low"),
      Stream.source(list(observations["close"]), dtype="float").rename("close")
      #Stream.source(list(data["volume"]), dtype="float").rename("volume") 
  ])

  reward_scheme = default.rewards.SimpleProfit()
  action_scheme = default.actions.SimpleOrders(trade_sizes=1)
  '''
  # define reward-scheme
  # define action-scheme
  action_scheme = default.actions.BSH(
      cash=cash,
      asset=asset
  )
  '''

  # create env
  env = default.create(
      portfolio=portfolio,
      action_scheme=action_scheme,
      reward_scheme=reward_scheme,
      feed=feed,
      renderer_feed=renderer_feed,
      #renderer="screen-log",
      renderer=default.renderers.PlotlyTradingChart(),
      #window_size=20,
      max_allowed_loss=0.6
  )

  return env

def evaluate_model(predict_fn, env, 
                  indicator_symbol: str,
                  num_steps=1000):
  """
  Evaluate a RL agent
  :param predict_fn: predict function
  :param env: Trading-Env to be used
  :param num_steps: (int) number of timesteps to evaluate it
  :return: (float) Mean reward for the last 100 episodes
  """
  episode_rewards = [0.0]
  obs = env.reset()
  done = False
  while not done:
      # _states are only useful when using LSTM policies
      action, _states = predict_fn(obs,indicator_symbol=indicator_symbol), None
      obs, reward, done, info = env.step(action)
      # Stats
      episode_rewards[-1] += reward
  env.render()

  # Compute mean reward for the last 100 edpisodes
  mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
  print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
  
  performance = pd.DataFrame.from_dict(env.action_scheme.portfolio.performance, orient='index')
  performance['net_worth'].plot()  
  return mean_100ep_reward

# Here the expert is a random agent
# but it can be any python function, e.g. a PID controller
def expert_trader(_obs, debug_info:bool = False):
    """
    Random agent. It samples actions randomly
    from the action space of the environment.

    :param _obs: (np.ndarray) Current observation
    :return: (np.ndarray) action taken by the expert
    """
    global df_min_ts
    global df_max_ts
    global global_last_action
    global global_buy_counter
    global global_sell_counter

    if debug_info:
      print("obs:=", _obs[0][0],_obs[0][1],_obs[0][2],_obs[0][3])

    # use df_min_ts.iloc[:, 1] to access columns by indices to match observations arrays
    is_buy_action = not (df_min_ts.loc[(df_min_ts.iloc[:, 0] == _obs[0][0]) & 
           (df_min_ts.iloc[:, 1] == _obs[0][1])  &
           (df_min_ts.iloc[:, 2] == _obs[0][2])  &
           (df_min_ts.iloc[:, 3] == _obs[0][3])
    ].empty)

    is_sell_action = not (df_max_ts.loc[(df_max_ts.iloc[:, 0] == _obs[0][0]) & 
           (df_max_ts.iloc[:, 1] == _obs[0][1])  &
           (df_max_ts.iloc[:, 2] == _obs[0][2])  &
           (df_max_ts.iloc[:, 3] == _obs[0][3])
        ].empty)

    if is_buy_action:
        #perform buy action
        global_last_action = 1
        global_buy_counter += 1
        if debug_info:
          print("buy-action",global_buy_counter)
    elif is_sell_action:
        #perform sell action
        global_last_action = 0
        global_sell_counter += 1
        if debug_info:
          print("sell-action",global_sell_counter)
    else:
        #do nothing
        pass

    return global_last_action

# %% [markdown]
# ## Expert DataSet


# %% [markdown]
#  # Trading Data

# %%
symbol = 'AAPL'
exchange = 'NASDAQ'
start_date = '2010-01-01'
end_date = '2020-12-31'

quotes = download_data(symbol=symbol, start_date=start_date, end_date=end_date, plot=True)
quotes.head()

# %% [markdown]
# ## Apply Technical-Indicators (TA)
# - Check https://github.com/bukosabino/ta
# - TA- Visualization: https://github.com/bukosabino/ta/blob/master/examples_to_use/visualize_features.ipynb

# %%
# get ta-indicators
from utils import TaFeatures
ta_f = TaFeatures(features=['MACD', 'BB', 'RSI'])
#data = ta_f.add_indicators(quotes,fillna=True)
#data = add_custom_ta_features(quotes,"open","high","low","close", fillna=True,plot=False,apply_pct=False)


# %% [markdown]
# # Create Trading-Enviornment

# %%
env = create_trade_env(quotes, quotes,symbol)
env.observer.feed.next()

# %%
from utils import TaFeatures
ta_f = TaFeatures(features=['MACD', 'BB', 'RSI'])
indicator_symbol = 'MACD' # in ['EMA','SMA','RSI']:
env = create_trade_env(quotes, quotes,symbol)
env.observer.feed.next()
evaluate_model(predict_fn=ta_f.predict,
    indicator_symbol=indicator_symbol,
    env=env)


# %%
indicator_symbol = 'MACD' # in ['EMA','SMA','RSI']:
df = ta_f.get_indicator_signals(quotes=quotes,
                            indicator_symbol=indicator_symbol,
                            fillna=True)
df = ta_f.get_BSH_signals(df=df,
indicator_symbol=indicator_symbol, 
                            plot=True)

# %%
# Reference: Python for Finance Cookbook - Over 50 Recei...
# Page: 210
import pyfolio as pf
RISKY_ASSETS = ['AAPL']
START_DATE = '2017-01-01'
END_DATE = '2020-12-31'
n_assets = len(RISKY_ASSETS)
#3. Download the stock prices from Yahoo Finance:
prices_df = yf.download(RISKY_ASSETS, start=START_DATE,
end=END_DATE, adjusted=True)
#4. Calculate individual asset returns:
returns = prices_df['Adj Close'].pct_change().dropna()
#5. Define the weights:
portfolio_weights = n_assets * [1 / n_assets]
#6. Calculate the portfolio returns:
#portfolio_returns = pd.Series(np.dot(portfolio_weights, returns.T),
#index=returns.index)
portfolio_returns = returns
#7. Create the tear sheet (simple variant):
pf.create_simple_tear_sheet(portfolio_returns)



# %%
data = ta_f.get_BSH_signals(df=data, indicator_symbol='SRSI', plot=True)

# %% [markdown]
# ### Plot Technical Indicators

# %% 


# %% 
# Plot BB
ta_f.plot(df=data, indicator_symbol='BB')

# %% RSI
ta_f.plot(df=data, indicator_symbol='RSI')

# %% [markdown]
# ### Get Buy/Sell Signals

# %%
ta_f.plot(df=data, indicator_symbol='RSI')

# %% [markdown]
# ## Get Local Minima/Maxima
# 

# %%
# get Min/Max TimeStamps
tmp_data = data.iloc[:,0:4]
tmp_data['org_close'] = quotes['close']
df_min_ts, df_max_ts = find_loc_min_max(data=tmp_data,order_of_points=7, plot=True)
df_min_ts.head()

# %%
quotes.head()


# %% [markdown]
# # Generate Expert Records

# %% [markdown]
# # Read Recording Set

# %%
# Pre-Train a Model using Behavior Cloning
#import ExpertDataset
# Using only one expert trajectory
# you can specify `traj_limitation=-1` for using the whole dataset
from utils import CustomExpertDataset
dataset = CustomExpertDataset(expert_path='expert_trader_ORG_'+ symbol +'.npz',
                        traj_limitation=10, batch_size=64, randomize = False)
dataset.plot()
print(dataset.observations.shape)

# %% [markdown]
# # Train RL-Agent using Expert-Records

# %%
# PPO2-Model
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy

VecEnv = DummyVecEnv([lambda: create_trade_env(quotes, data,symbol)])
agent = PPO2(MlpPolicy, env=VecEnv, verbose=1,tensorboard_log=os.path.join(currentdir,"logs"))
# Pretrain the PPO2 model
#agent.pretrain(dataset, n_epochs=1000)

# As an option, you can train the RL agent
agent.learn(int(1e3),tb_log_name="learn_"+symbol)
agent.save(save_path=os.path.join(currentdir, "BC_PPO2_MlpPolicy_NORM.zip"))


# %%
# Load the TensorBoard notebook extension
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir logs/')
#tensorboard = TensorBoard(log_dir="./logs")

# %% [markdown]
#  ## Evaluate Model

# %%
symbol = 'AACQU'
start_date = '2010-01-01'
end_date = '2020-12-11'
#MSFT, TSLA, AAPL,NFLX,GOOG, GLD
quotes = download_data(symbol=symbol, start_date=start_date, end_date=end_date,plot=True)
data = add_custom_ta_features(quotes,"open","high","low","close", fillna=True)
#df_min_ts, df_max_ts = find_loc_min_max(data=quotes,order_of_points=7, plot=True)
env = create_trade_env(quotes, data,symbol)


# %%
# %%
VecEnv = DummyVecEnv([lambda: create_trade_env(quotes, data,symbol)])
agent = PPO2.load(load_path=os.path.join(currentdir, "BC_PPO2_MlpPolicy_NORM.zip"))
#agent = DQN.load(load_path=os.path.join(currentdir, "agents","DQN_MlpPolicy_02.zip"), env=env)
evaluate_model(agent, env)




# %% [markdown]
# # Load Financial Symbols

# %%
get_ipython().system('pip install finsymbols')


# %%
from finsymbols import symbols
import json
import pprint

#symbol_list = symbols.get_sp500_symbols()
#symbol_list.extend(symbols.get_amex_symbols())
#symbol_list.extend(symbols.get_nyse_symbols())
#symbol_list.extend(symbols.get_nasdaq_symbols())
symbol_list = symbols.get_nasdaq_symbols()

column_names = ['company','headquarters', 'industry','sector','symbol']
df = pd.DataFrame(symbol_list, columns=column_names)
my_symbols = df['symbol'].replace("\n", "", regex=True)

# %% [markdown]
# # Loops
# %% [markdown]
# ## Create expert Recordings

# %%
# Download List of NASDAQ Insturment
df = pd.read_csv('nasdaq_list.csv')
#df = df.iloc[17:]
df.head()


# %%
start_date = '2010-01-01'
end_date = '2020-12-11'
for symbol in df['Symbol']:
  #MSFT, TSLA, AAPL,NFLX,GOOG, GLD
  print("symbol:=", symbol)
  quotes = download_data(symbol=symbol, start_date=start_date, end_date=end_date,plot=True)
  if (not quotes.empty) and (len(quotes)>100):
    data = add_custom_ta_features(quotes,"open","high","low","close", fillna=True)
    # get Min/Max TimeStamps
    tmp_data = data.iloc[:,0:4]
    tmp_data['org_close'] = quotes['close']
    df_min_ts, df_max_ts = find_loc_min_max(data=tmp_data,order_of_points=7, plot=True, symbol=symbol)
    env = create_trade_env(quotes, data,symbol)

    global_buy_counter = 0
    global_sell_counter = 0
    global_last_action = 0
    try:
      generate_expert_traj(expert_trader, 'expert_trader_'+symbol, env, n_episodes=10)
    except:
      print("An exception occurred while generating recording for symbol:=",symbol)
    

# %% [markdown]
# ## Trainning Loop

# %%
current = os.getcwd()
model_path = os.path.join(currentdir, "LOOP_PPO2_MlpPolicy_NORM.zip")

for filename in os.listdir(current):
    #extract pretrain file
    if filename.endswith(".npz"):
      # get symbol-name
      x = filename.split("expert_trader_")
      x= x[1].split(".npz")
      symbol=x[0]

      f = open('traing_progress.txt', 'a')
      f.write("pre-train: " + symbol)
      f.close()
      
      # create env
      quotes = download_data(symbol=symbol, start_date=start_date, end_date=end_date,plot=True)
      data = add_custom_ta_features(quotes,"open","high","low","close", fillna=True)
      env = create_trade_env(quotes, data,symbol)
      VecEnv = DummyVecEnv([lambda: create_trade_env(quotes, data,symbol)])

      if os.path.isfile(model_path):
        #load agent
        agent = PPO2.load(load_path=model_path, env=VecEnv,tensorboard_log=os.path.join(currentdir,"logs"))
        print("Agent has been loaded: Symbol= ", symbol)
        
      else:
        #create new agent
        agent = PPO2(policy=MlpPolicy, env=VecEnv, verbose=1,tensorboard_log=os.path.join(currentdir,"logs"))
        print("new Agent has been created: Symbol= ", symbol)

      # Pretrain the PPO2 model
      dataset = ExpertDataset(expert_path='expert_trader_'+ symbol +'.npz',
                              traj_limitation=10, batch_size=64, randomize = False)      
      agent.pretrain(dataset, n_epochs=100)

      # As an option, you can train the RL agent
      agent.learn(int(1e4),tb_log_name="learn_"+symbol)
      
      #save Model
      agent.save(save_path=model_path)
      print("Agent has been Saved: Symbol= ", symbol)
      print("--------------------------------------------------")


    else:
        continue


# %%
VecEnv = DummyVecEnv([lambda: create_trade_env(quotes, data,symbol)])

#agent = DQN.load(load_path=os.path.join(currentdir, "agents","DQN_MlpPolicy_02.zip"), env=env)
evaluate_model(agent, env)


# %%
agent = PPO2.load(load_path=os.path.join(currentdir, "BC_PPO2_MlpPolicy_NORM.zip"))
# Pretrain the PPO2 model
agent.pretrain(dataset, n_epochs=1000)

# As an option, you can train the RL agent
agent.learn(int(1e5),tb_log_name="learn_"+symbol)

# %% [markdown]
# # Evaulate using Pyfolio

# %%
rets = px[['AdjClose']]
rets = rets.shift(-1)
rets.iloc[-1]['AdjClose'] = px.tail(1)['AdjOpen']
rets = rets.shift(1) / rets - 1
rets = rets.dropna()
rets.index = rets.index.to_datetime()
rets.index = rets.index.tz_localize("UTC")
rets.columns = [symbol]
return rets


