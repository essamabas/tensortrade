# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
#  # Install Stable-baselines/ TensorTrade - Colab

# %%
#install stable-baselines
get_ipython().system('sudo apt-get update && sudo apt-get install cmake libopenmpi-dev zlib1g-dev')

# setup dependencies
# !python3 -m pip install git+https://github.com/tensortrade-org/tensortrade.git
get_ipython().system('python3 -m pip install git+https://github.com/essamabas/tensortrade.git@live')
get_ipython().system('pip install yfinance ta matplotlib s3fs')


# %%
get_ipython().system('pip install stable-baselines[mpi]==2.10.1')
#select tensorflow version 1. - 
get_ipython().run_line_magic('tensorflow_version', '1.x')


# %%
import stable_baselines
stable_baselines.__version__


# %% [markdown]
# # Include Libraries

# %%
# setup dependencies
import inspect
import sys
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
#sys.path.insert(0, "{}".format(parentdir))
sys.path.append(parentdir)
currentdir


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
from datetime import datetime

from scipy.signal import argrelextrema
import numpy as np

import yfinance as yf

from plotly.subplots import make_subplots
import plotly.graph_objects as go

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
                  end_date: str = datetime.today().strftime('%Y-%m-%d'),
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

## Apply Technical-Indicators (TA)
#- Check https://github.com/bukosabino/ta
#- TA- Visualization: https://github.com/bukosabino/ta/blob/master/examples_to_use/visualize_features.ipynb
def add_custom_ta_features(
    df: pd.DataFrame,
    open: str,  # noqa
    high: str,
    low: str,
    close: str,
    fillna: bool = False,
    colprefix: str = "",
    apply_pct: bool = False,
    plot: bool = False,
) -> pd.DataFrame:

    # Add Volatility TA
    df = ta.add_volatility_ta(
        df=df, high=high, low=low, close=close, fillna=fillna, colprefix=colprefix
    )
    # Add Trend TA
    df = ta.add_trend_ta(
        df=df, high=high, low=low, close=close, fillna=fillna, colprefix=colprefix
    )
    # Add Other TA
    df = ta.add_others_ta(df=df, close=close, fillna=fillna, colprefix=colprefix)

    # convert to pct
    if apply_pct:
      df = df.pct_change(fill_method ='ffill')
      df = df.applymap(lambda x: x*100)
      df.replace([np.inf, -np.inf], np.nan,inplace=True)
    df.astype(np.float32)    
    df = df.round(5)

    if fillna: 
      df.fillna(value=0,inplace=True)

    if plot:
      fig = make_subplots(rows=5, cols=1,
                          shared_xaxes=True,
                          vertical_spacing=0.02,
                          subplot_titles=("Close", "Bollinger Bands","MACD"))

      fig.add_trace(go.Scatter(
          x=df.index,
          y=df['close'],
          name = symbol
      ), row=1, col=1)

      # Bollinger-Bands
      fig.add_trace(go.Scatter(
          x=df.index,
          y=df['close'],
          name = symbol
      ), row=2, col=1)

      fig.add_trace(go.Scatter(
          x=df.index,
          y=df['volatility_bbh'],
          name = symbol+' High BB'
      ), row=2, col=1)

      fig.add_trace(go.Scatter(
          x=df.index,
          y=df['volatility_bbl'],
          name = symbol+' Low BB'
      ), row=2, col=1)

      fig.add_trace(go.Scatter(
          x=df.index,
          y=df['volatility_bbm'],
          name = symbol+' EMA BB' 
      ), row=2, col=1)

      # MACD
      fig.add_trace(go.Scatter(
          x=df.index,
          y=df['trend_macd'],
          name = symbol+' MACD'
      ), row=3, col=1)

      fig.add_trace(go.Scatter(
          x=df.index,
          y=df['trend_macd_signal'],
          name = symbol+' MACD Signal'
      ), row=3, col=1)

      fig.add_trace(go.Scatter(
          x=df.index,
          y=df['trend_macd_diff'],
          name = symbol+' MACD Difference'
      ), row=3, col=1)

      # SMA
      fig.add_trace(go.Scatter(
          x=df.index,
          y=df['close'],
          name = symbol
      ), row=4, col=1)

      fig.add_trace(go.Scatter(
          x=df.index,
          y=df['trend_sma_fast'],
          name = symbol+' SMA-Fast'
      ), row=4, col=1)

      fig.add_trace(go.Scatter(
          x=df.index,
          y=df['trend_sma_slow'],
          name = symbol+' SMA-Slow'
      ), row=4, col=1)

      # EMA
      fig.add_trace(go.Scatter(
          x=df.index,
          y=df['close'],
          name = symbol
      ), row=5, col=1)

      fig.add_trace(go.Scatter(
          x=df.index,
          y=df['trend_ema_fast'],
          name = symbol+' EMA-Fast'
      ), row=5, col=1)

      fig.add_trace(go.Scatter(
          x=df.index,
          y=df['trend_ema_slow'],
          name = symbol+' EMA-Slow'
      ), row=5, col=1)              

      config = {'displayModeBar': False}
      fig.show(config=config)
     
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
  for c in data.columns[0:]:
      s = Stream.source(list(data[c]), dtype="float").rename(data[c].name)
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
      Stream.source(list(data.index)).rename("date"),
      Stream.source(list(data["open"]), dtype="float").rename("open"),
      Stream.source(list(data["high"]), dtype="float").rename("high"),
      Stream.source(list(data["low"]), dtype="float").rename("low"),
      Stream.source(list(data["close"]), dtype="float").rename("close")
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
      #window_size=20,
      max_allowed_loss=0.6
  )

  return env

def evaluate_model(model, env, num_steps=1000):
  """
  Evaluate a RL agent
  :param model: (BaseRLModel object) the RL Agent
  :param env: Trading-Env to be used
  :param num_steps: (int) number of timesteps to evaluate it
  :return: (float) Mean reward for the last 100 episodes
  """
  episode_rewards = [0.0]
  obs = env.reset()
  done = False
  while not done:
      # _states are only useful when using LSTM policies
      action, _states = model.predict(obs)
      obs, reward, done, info = env.step(action)
      # Stats
      episode_rewards[-1] += reward

  # Compute mean reward for the last 100 episodes
  mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
  print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
  
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

# %%
# %%
import queue
import time
from multiprocessing import Queue, Process

import cv2  # pytype:disable=import-error
import numpy as np
from joblib import Parallel, delayed
from stable_baselines import logger


class ExpertDataset(object):
    """
    Dataset for using behavior cloning or GAIL.

    The structure of the expert dataset is a dict, saved as an ".npz" archive.
    The dictionary contains the keys 'actions', 'episode_returns', 'rewards', 'obs' and 'episode_starts'.
    The corresponding values have data concatenated across episode: the first axis is the timestep,
    the remaining axes index into the data. In case of images, 'obs' contains the relative path to
    the images, to enable space saving from image compression.

    :param expert_path: (str) The path to trajectory data (.npz file). Mutually exclusive with traj_data.
    :param traj_data: (dict) Trajectory data, in format described above. Mutually exclusive with expert_path.
    :param train_fraction: (float) the train validation split (0 to 1)
        for pre-training using behavior cloning (BC)
    :param batch_size: (int) the minibatch size for behavior cloning
    :param traj_limitation: (int) the number of trajectory to use (if -1, load all)
    :param randomize: (bool) if the dataset should be shuffled
    :param verbose: (int) Verbosity
    :param sequential_preprocessing: (bool) Do not use subprocess to preprocess
        the data (slower but use less memory for the CI)
    """
    # Excluded attribute when pickling the object
    EXCLUDED_KEYS = {'dataloader', 'train_loader', 'val_loader'}

    def __init__(self, expert_path=None, traj_data=None, train_fraction=0.7, batch_size=64,
                 traj_limitation=-1, randomize=True, verbose=1, sequential_preprocessing=False):
        if traj_data is not None and expert_path is not None:
            raise ValueError("Cannot specify both 'traj_data' and 'expert_path'")
        if traj_data is None and expert_path is None:
            raise ValueError("Must specify one of 'traj_data' or 'expert_path'")
        if traj_data is None:
            traj_data = np.load(expert_path, allow_pickle=True)

        if verbose > 0:
            for key, val in traj_data.items():
                print(key, val.shape)

        # Array of bool where episode_starts[i] = True for each new episode
        episode_starts = traj_data['episode_starts']

        traj_limit_idx = len(traj_data['obs'])

        if traj_limitation > 0:
            n_episodes = 0
            # Retrieve the index corresponding
            # to the traj_limitation trajectory
            for idx, episode_start in enumerate(episode_starts):
                n_episodes += int(episode_start)
                if n_episodes == (traj_limitation + 1):
                    traj_limit_idx = idx - 1

        observations = traj_data['obs'][:traj_limit_idx]
        actions = traj_data['actions'][:traj_limit_idx]

        # obs, actions: shape (N * L, ) + S
        # where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # S = (1, ) for discrete space
        # Flatten to (N * L, prod(S))
        if len(observations.shape) > 2:
            #observations = np.reshape(observations, [-1, np.prod(observations.shape[1:])])
            pass
        if len(actions.shape) > 2:
            #actions = np.reshape(actions, [-1, np.prod(actions.shape[1:])])
            pass

        indices = np.random.permutation(len(observations)).astype(np.int64)

        # Train/Validation split when using behavior cloning
        train_indices = indices[:int(train_fraction * len(indices))]
        val_indices = indices[int(train_fraction * len(indices)):]

        assert len(train_indices) > 0, "No sample for the training set"
        assert len(val_indices) > 0, "No sample for the validation set"

        self.observations = observations
        self.actions = actions

        self.returns = traj_data['episode_returns'][:traj_limit_idx]
        self.avg_ret = sum(self.returns) / len(self.returns)
        self.std_ret = np.std(np.array(self.returns))
        self.verbose = verbose

        assert len(self.observations) == len(self.actions), "The number of actions and observations differ "                                                             "please check your expert dataset"
        self.num_traj = min(traj_limitation, np.sum(episode_starts))
        self.num_transition = len(self.observations)
        self.randomize = randomize
        self.sequential_preprocessing = sequential_preprocessing

        self.dataloader = None
        self.train_loader = DataLoader(train_indices, self.observations, self.actions, batch_size,
                                       shuffle=self.randomize, start_process=False,
                                       sequential=sequential_preprocessing)
        self.val_loader = DataLoader(val_indices, self.observations, self.actions, batch_size,
                                     shuffle=self.randomize, start_process=False,
                                     sequential=sequential_preprocessing)

        if self.verbose >= 1:
            self.log_info()

    def init_dataloader(self, batch_size):
        """
        Initialize the dataloader used by GAIL.

        :param batch_size: (int)
        """
        indices = np.random.permutation(len(self.observations)).astype(np.int64)
        self.dataloader = DataLoader(indices, self.observations, self.actions, batch_size,
                                     shuffle=self.randomize, start_process=False,
                                     sequential=self.sequential_preprocessing)

    def __del__(self):
        # Exit processes if needed
        for key in self.EXCLUDED_KEYS:
            if self.__dict__.get(key) is not None:
                del self.__dict__[key]

    def __getstate__(self):
        """
        Gets state for pickling.

        Excludes processes that are not pickleable
        """
        # Remove processes in order to pickle the dataset.
        return {key: val for key, val in self.__dict__.items() if key not in self.EXCLUDED_KEYS}

    def __setstate__(self, state):
        """
        Restores pickled state.

        init_dataloader() must be called
        after unpickling before using it with GAIL.

        :param state: (dict)
        """
        self.__dict__.update(state)
        for excluded_key in self.EXCLUDED_KEYS:
            assert excluded_key not in state
        self.dataloader = None
        self.train_loader = None
        self.val_loader = None

    def log_info(self):
        """
        Log the information of the dataset.
        """
        logger.log("Total trajectories: {}".format(self.num_traj))
        logger.log("Total transitions: {}".format(self.num_transition))
        logger.log("Average returns: {}".format(self.avg_ret))
        logger.log("Std for returns: {}".format(self.std_ret))

    def get_next_batch(self, split=None):
        """
        Get the batch from the dataset.

        :param split: (str) the type of data split (can be None, 'train', 'val')
        :return: (np.ndarray, np.ndarray) inputs and labels
        """
        dataloader = {
            None: self.dataloader,
            'train': self.train_loader,
            'val': self.val_loader
        }[split]

        if dataloader.process is None:
            dataloader.start_process()
        try:
            return next(dataloader)
        except StopIteration:
            dataloader = iter(dataloader)
            return next(dataloader)

    def plot(self):
        """
        Show histogram plotting of the episode returns
        """
        # Isolate dependency since it is only used for plotting and also since
        # different matplotlib backends have further dependencies themselves.
        import matplotlib.pyplot as plt
        plt.hist(self.returns)
        plt.show()


class DataLoader(object):
    """
    A custom dataloader to preprocessing observations (including images)
    and feed them to the network.

    Original code for the dataloader from https://github.com/araffin/robotics-rl-srl
    (MIT licence)
    Authors: Antonin Raffin, René Traoré, Ashley Hill

    :param indices: ([int]) list of observations indices
    :param observations: (np.ndarray) observations or images path
    :param actions: (np.ndarray) actions
    :param batch_size: (int) Number of samples per minibatch
    :param n_workers: (int) number of preprocessing worker (for loading the images)
    :param infinite_loop: (bool) whether to have an iterator that can be reset
    :param max_queue_len: (int) Max number of minibatches that can be preprocessed at the same time
    :param shuffle: (bool) Shuffle the minibatch after each epoch
    :param start_process: (bool) Start the preprocessing process (default: True)
    :param backend: (str) joblib backend (one of 'multiprocessing', 'sequential', 'threading'
        or 'loky' in newest versions)
    :param sequential: (bool) Do not use subprocess to preprocess the data
        (slower but use less memory for the CI)
    :param partial_minibatch: (bool) Allow partial minibatches (minibatches with a number of element
        lesser than the batch_size)
    """

    def __init__(self, indices, observations, actions, batch_size, n_workers=1,
                 infinite_loop=True, max_queue_len=1, shuffle=False,
                 start_process=True, backend='threading', sequential=False, partial_minibatch=True):
        super(DataLoader, self).__init__()
        self.n_workers = n_workers
        self.infinite_loop = infinite_loop
        self.indices = indices
        self.original_indices = indices.copy()
        self.n_minibatches = len(indices) // batch_size
        # Add a partial minibatch, for instance
        # when there is not enough samples
        if partial_minibatch and len(indices) % batch_size > 0:
            self.n_minibatches += 1
        self.batch_size = batch_size
        self.observations = observations
        self.actions = actions
        self.shuffle = shuffle
        self.queue = Queue(max_queue_len)
        self.process = None
        self.load_images = isinstance(observations[0], str)
        self.backend = backend
        self.sequential = sequential
        self.start_idx = 0
        if start_process:
            self.start_process()

    def start_process(self):
        """Start preprocessing process"""
        # Skip if in sequential mode
        if self.sequential:
            return
        self.process = Process(target=self._run)
        # Make it a deamon, so it will be deleted at the same time
        # of the main process
        self.process.daemon = True
        self.process.start()

    @property
    def _minibatch_indices(self):
        """
        Current minibatch indices given the current pointer
        (start_idx) and the minibatch size
        :return: (np.ndarray) 1D array of indices
        """
        return self.indices[self.start_idx:self.start_idx + self.batch_size]

    def sequential_next(self):
        """
        Sequential version of the pre-processing.
        """
        if self.start_idx > len(self.indices):
            raise StopIteration

        if self.start_idx == 0:
            if self.shuffle:
                # Shuffle indices
                np.random.shuffle(self.indices)

        obs = self.observations[self._minibatch_indices]
        if self.load_images:
            obs = np.concatenate([self._make_batch_element(image_path) for image_path in obs],
                                 axis=0)

        actions = self.actions[self._minibatch_indices]
        self.start_idx += self.batch_size
        return obs, actions

    def _run(self):
        start = True
        with Parallel(n_jobs=self.n_workers, batch_size="auto", backend=self.backend) as parallel:
            while start or self.infinite_loop:
                start = False

                if self.shuffle:
                    np.random.shuffle(self.indices)

                for minibatch_idx in range(self.n_minibatches):

                    self.start_idx = minibatch_idx * self.batch_size

                    obs = self.observations[self._minibatch_indices]
                    if self.load_images:
                        if self.n_workers <= 1:
                            obs = [self._make_batch_element(image_path)
                                   for image_path in obs]

                        else:
                            obs = parallel(delayed(self._make_batch_element)(image_path)
                                           for image_path in obs)

                        obs = np.concatenate(obs, axis=0)

                    actions = self.actions[self._minibatch_indices]

                    self.queue.put((obs, actions))

                    # Free memory
                    del obs

                self.queue.put(None)

    @classmethod
    def _make_batch_element(cls, image_path):
        """
        Process one element.

        :param image_path: (str) path to an image
        :return: (np.ndarray)
        """
        # cv2.IMREAD_UNCHANGED is needed to load
        # grey and RGBa images
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # Grey image
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]

        if image is None:
            raise ValueError("Tried to load {}, but it was not found".format(image_path))
        # Convert from BGR to RGB
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((1,) + image.shape)
        return image

    def __len__(self):
        return self.n_minibatches

    def __iter__(self):
        self.start_idx = 0
        self.indices = self.original_indices.copy()
        return self

    def __next__(self):
        if self.sequential:
            return self.sequential_next()

        if self.process is None:
            raise ValueError("You must call .start_process() before using the dataloader")
        while True:
            try:
                val = self.queue.get_nowait()
                break
            except queue.Empty:
                time.sleep(0.001)
                continue
        if val is None:
            raise StopIteration
        return val

    def __del__(self):
        if self.process is not None:
            self.process.terminate()

# %% [markdown]
#  # Trading Data

# %%
symbol = 'AAPL'
exchange = 'NASDAQ'
start_date = '2010-01-01'
end_date = '2020-12-11'

quotes = download_data(symbol=symbol, start_date=start_date, end_date=end_date, plot=True)
quotes.head()

# %% [markdown]
# ## Apply Technical-Indicators (TA)
# - Check https://github.com/bukosabino/ta
# - TA- Visualization: https://github.com/bukosabino/ta/blob/master/examples_to_use/visualize_features.ipynb

# %%
# get ta-indicators
data = add_custom_ta_features(quotes,"open","high","low","close", fillna=True,plot=True,apply_pct=False)
data.tail()

# %% [markdown]
# ## Get Local Minima/Maxima
# 

# %%
# get Min/Max TimeStamps
tmp_data = data.iloc[:,0:4]
tmp_data['org_close'] = quotes['close']
df_min_ts, df_max_ts = find_loc_min_max(data=tmp_data,order_of_points=7, plot=True)
df_min_ts.head()


# %% [markdown]
# # Create Trading-Enviornment

# %%
env = create_trade_env(quotes, data,symbol)


# %%
env.observer.feed.next()


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
agent.learn(int(1e5),tb_log_name="learn_"+symbol)
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


# %%
#portfolio.performance.net_worth.plot()
performance = pd.DataFrame.from_dict(env.action_scheme.portfolio.performance, orient='index')
performance['net_worth'].plot()


# %%
performance['net_worth'].tail()

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


