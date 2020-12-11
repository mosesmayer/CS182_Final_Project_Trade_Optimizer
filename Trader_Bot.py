import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qlearning_functions as simulation_cmd

class Stockinfo:
    def __init__(self, ticker, csvpath):
        self.ticker = ticker
        self.filepath = csvpath
    
    def __repr__(self):
        return f"{self.ticker} data from {self.filepath}"

class Trader:
    """
    Trader and optimizer bot. Discretizes stock data into discrete states and
    trains using Q-learning to determine optimal trades.
    
    FUNCTIONS:
        reset()
        compute_EMA(series, num_days)
        compute_MACD_signal(series)
        compute_stochastic_oscillators(series)
        compute_high_low(series, back_days)
        compute_MA(series, lag)
        map_to_states(series, state_ranges)
        compute_states(series, k)
        choose_action(state, ticker, epsilon)
        simulate_trader(df, timeframe, validate, print_dates, print_stats, graph_stats)
    """

    def __init__(self, **kwargs):
        """
        Initial constructor. Pass in dictionary mapping stock names to data\
        filepaths under keyword "stocks".

        PARAMETERS:
            stocks (dict)
                a dictionary that maps stock tickers to the csv file containing
                historical data.
            balance (float)
                initial balance in the portfolio
            epsilon, alpha, gamma (floats)
                hyperparameters
            state_model (str)
                model to use in order to initialize states
        """
        self.default_values = {
            "stocks": {},
            "balance": 100000.0,
            "epsilon": 0.1,
            "alpha": 0.1,
            "gamma": 1.0,
            "state_model": "ST_tau"
        }
        print("\n===========INITIALIZING===========\n")
        for key in self.default_values.keys():
            if (key not in kwargs.keys()):
                print(f"Argument {key} not found, assuming default value of {self.default_values[key]}")
            else:
                print(f"Using {kwargs[key].__repr__()} for key \"{key}\"")
                self.default_values[key] = kwargs[key]

        self.data = {} # loaded dataframes for each stock

        train_test_split = 0.7 # split for training and testing data
        self.train_set = {}
        self.test_set = {}

        self.state_ranges = {}

        self.portfolio = {'balance':self.default_values["balance"]}
        self.transaction_volume = 1000 # assumption: transact a fixed number of shares each time.
        self.commission = 0.0012 # commission on trades.
        self.buys = {} # number of active buy actions for each ticker.
        
        self.q_table = {}
        self.progress_profits = {} # keep track of annual profits of episodes
        self.progress_rewards = {} # keep track of annual rewards of episodes

        # hyperparameters
        self.epsilon = self.default_values["epsilon"]
        self.alpha = self.default_values["alpha"]
        self.gamma = self.default_values["gamma"]

        max_years_ = []
        for stockdata in self.default_values["stocks"]:
            ticker = stockdata.ticker
            path = stockdata.filepath
            df = pd.read_csv(path, usecols=['Date', 'Close', 'Open']).sort_values('Date')
            df['delta'] = df.Close.pct_change()
            df['EMA'] = self.compute_EMA(df.Close)
            df['LT_MA'] = self.compute_MA(df.Close, lag = 200)
            df['ST_MA'] = self.compute_MA(df.Close, lag = 50)
            df['MACD'], df['signal_line'] = self.compute_MACD_signal(df.Close)
            df['ST_tau'] = (df.Close - df.ST_MA)
            df['LT_tau'] = (df.Close - df.LT_MA)
            df['Stochastic'] = self.compute_stochastic_oscillators(df.Close)

            # Get states
            df['state'], self.state_ranges[ticker] = self.compute_states(df[self.default_values["state_model"]])

            # Get data
            temp = df.dropna().reset_index(drop=True)
            max_years_.append(float(len(temp)) / len(df) * 7.0) # establish maximum data length
            self.data[ticker] = temp.copy()
            train_test_limit = int(train_test_split * len(temp))
            self.train_set[ticker] = temp.iloc[:train_test_limit].reset_index(drop=True)
            self.test_set[ticker] = temp.iloc[train_test_limit:].reset_index(drop=True)

            # Initialize q_table and progress arrays
            self.q_table[ticker] = {}
            self.progress_profits[ticker] = []
            self.progress_rewards[ticker] = []

            # setup a starting portfolio for the trader
            self.portfolio[ticker] = {'holdings': 0}

            self.buys[ticker] = 0.0
        self.max_years = min(max_years_) # get lowest max_years among all datasets

    def reset(self):
        """ Resets trader's portfolio for another run of the simulation """
        for key in self.portfolio.keys():
            self.portfolio[key] = {'holdings': 0}
            self.buys[key] = 0
        self.portfolio['balance'] = self.default_values['balance']
            
    def compute_EMA(self, series, num_days = 50):
        """
        Computes Exponential Moving Averages of data.
        
        PARAMETERS:
            series (pandas Series)
                timeseries data
            num_days (int, default=50)
                the smoothing period
        
        RETURN VALUE:
            A pandas Series containing EMA's for every index of original series.
        """
        temp = series.copy().reset_index(drop=True)
        smoothing_factor = 2
        smoothing_ratio = smoothing_factor/(num_days+1)
        prevEMA = 0.0
        for i in range(len(temp)):
            currEMA = (temp[i] * smoothing_ratio) + prevEMA * (1 - smoothing_ratio)
            temp[i] = currEMA
            prevEMA = currEMA 
        return temp
    
    def compute_MACD_signal(self, series):
        """
        Computes Moving Average Convergence Divergence and signal line
        for entries in data.
        
        PARAMETERS:
            series (pandas Series)
                timeseries data to compute MACD
        
        RETURN VALUE:
            Two pandas series containing current values of MACD and signal line.
        """
        temp = series.copy().reset_index(drop=True)
        t1 = self.compute_EMA(temp, num_days=12)
        t2 = self.compute_EMA(temp, num_days=26)
        MACD = t1-t2
        signal_line = self.compute_EMA(MACD, num_days=9)
        return MACD, signal_line

    def compute_stochastic_oscillators(self, series):
        """
        Computes Stochastic Oscillators.

        PARAMETERS:
            series (pandas Series):
                timeseries data used to compute oscillators
        
        RETURN VALUE:
            A pandas Series containing the value of the stochastic indicator
            at given days.
        """
        temp = series.copy().reset_index(drop=True)
        high, low = self.compute_high_low(temp)
        K = (temp - low) / (high - low) * 100
        return K

    def compute_high_low(self, series, back_days = 14):
        """
        Computes high and lows over back_days periods.

        PARAMETERS:
            series (pandas Series):
                timeseries data used to compute oscillators
            back_days (int):
                number of days to compute extremes over
        
        RETURN VALUE:
            Two pandas Series containing the highs and lows.
        """
        highs = series.copy().reset_index(drop = True)
        lows = series.copy().reset_index(drop = True)
        assert len(highs) > back_days, 'Not enough datapoints for high/low!'
        for i in range(len(highs)):
            highs[i] = series[max(0, i-back_days):i].max()
            lows[i] = series[max(0, i-back_days):i].min()
        highs[:back_days] = None
        lows[:back_days] = None
        return highs, lows

    def compute_MA(self, series, lag = 50):
        """
        Computes long-term/short-term Moving Averages for the entries in 
        the data.
        
        PARAMETERS:
            series (pandas Series)
                timeseries data
            lag (int, default = 50)
                MA lag; number of days to compute average over. Use 50 for
                short term and 200 for long term.
        
        RETURN VALUE:
            A pandas Series containing the MA's for every row
        """
        temp = series.copy().reset_index(drop=True)
        assert len(temp) > lag, "Not enough points to generate MA's!"
        for idx in range(lag, len(temp)):
            temp[idx] = series[idx-lag:idx].mean()
        temp[:lag] = None
        return temp
    
    def map_to_states(self, series, state_ranges):
        """
        Maps values from series to discrete states according to ranges.
        
        PARAMETERS
            series (pandas Series)
                timeseries data
            state_ranges (dictionary)
                contains the percentile ranges for every state
        
        RETURNS
            A pandas Series containing the state for every row
        """
        states = series.copy().reset_index(drop=True)
        l_st = list(state_ranges.keys())[0] # lowest range index
        r_st = list(state_ranges.keys())[-1] # highest range index
        for idx, val in enumerate(series): # iterate through values in series
            # Check if current value is outside ranges of first and last percentile
            if val <= state_ranges[l_st][0]:
                states[idx] = l_st    
            elif val >= state_ranges[r_st][1]:
                states[idx] = r_st
            else: # In one of the predefined ranges 
                for state_idx, cur_range in enumerate(state_ranges):
                    l, r = cur_range[0], cur_range[1]
                    if l <= val and val < r:
                        states[idx] = state_idx
                        break
        return states
    
    def compute_states(self, series, k = 25):
        """
        Maps the values of series to discrete states using percentiles of series
        values.
        
        PARAMETERS:
            series (pandas Series)
                timeseries data
            k (int, default=True)
                number of states/percentile divisions
        
        RETURN VALUE:
            A pandas Series containing the state for every row.
        """
        # Compute percentile values
        levels = [i / k for i in range(1,k+1)]
        percentiles = pd.Series({i + 1 : series.quantile(val) for i, val in enumerate(levels)})
        # compute the ranges for each state using the percentiles
        percentiles = pd.Series(list(zip(percentiles.shift(periods = 1, fill_value=series.min()), percentiles)))
        # map values into states
        states = self.map_to_states(series, percentiles)
        return states, percentiles
 
    def choose_action(self, state, ticker, epsilon):
        """
        Chooses an action for the current state.
        
        PARAMETERS
            state (str)
                the current state
            ticker (str)
                the ticker of the company, used while searching transition information
            epsilon (float)
                the epsilon hyperparameter (randomness parameter)

        RETURNS
            A string equal to 'buy', 'sell', or 'hold'.
        """
        # For implementation see qlearning_functions.py file
        return simulation_cmd.choose_action(self, state, ticker, epsilon)
    
    def simulate_trader(self, df = None, timeframe = 1.0, validate=False, print_dates=False, print_stats=False, graph_stats=False):
        """
        Simulates the trader's actions on the data, employing the Q-learning algorithm.
        
        PARAMETERS
            df (dataframe)
                Contains relevant data (either training or testing dataset).
            timeframe (float)
                Stores how large of a time interval we use for each episode.
            validate (bool)
                Decides whether we are training or evaluating.
            print_dates (bool)
                Decides whether to print out the date spans.
            print_stats (bool)
                Decides whether to print out the simulation statistics.
            graph_stats (bool)
                Decides whether to display a graph of the buying and selling history for each stock and episode.
        """
        # For implementation see qlearning_functions.py file
        return simulation_cmd.simulate_trader(self, df, timeframe, validate, print_dates, print_stats, graph_stats)

if __name__ == "__main__":
    print("This file contains an implementation of our trading optimizer bot.")
    print("To run the bot, please refer to run.py.")