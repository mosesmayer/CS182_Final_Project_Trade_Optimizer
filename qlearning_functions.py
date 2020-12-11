import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    actions = ['buy', 'sell', 'hold']
    if state not in self.q_table[ticker]:
        self.q_table[ticker][state] = [0, 0, 0] # not in q-table yet
    if np.random.rand() < epsilon: # if below epsilon, employ exploration
        return np.random.choice(actions) # return random action
    a_index = np.argmax(self.q_table[ticker][state]) # otherwise, employ exploitation
    return actions[a_index] # return optimal action

def simulate_trader(self, df = None, timeframe = 1.0, validate=False, print_dates=False, print_stats=False, graph_stats=False):
    """
    Simulates the trader's actions on the data, employing the Q-learning algorithm.
    
    PARAMETERS
        df (dataframe)
            Contains relevant data (either training or testing dataset).
        timeframe (float)
            Stores how large of a time interval we use for each episode as a ratio of the length of df.
        validate (bool)
            Decides whether we are training or evaluating.
        print_dates (bool)
            Decides whether to print out the date spans.
        print_stats (bool)
            Decides whether to print out the simulation statistics.
        graph_stats (bool)
            Decides whether to display a graph of the buying and selling history for each stock and episode.
    """
    if timeframe > 1.0 or timeframe < 0:
        if not validate:
            print("time frame must be between [0, 1] (now {}), setting to 1".format(timeframe))
        timeframe = 1.0
    if not validate:
        print(f"Current time frame: {100 * timeframe}%")
    if df is None:
        df = self.data
        if not validate:
            print("Using all data")

    for name in df.keys(): # for each stock
        ticker_frame = df[name].copy() # get data

        total_length = len(ticker_frame)
        desired_length = min(int(timeframe * total_length), total_length)
        splits = [0]
        while splits[-1] + desired_length <= total_length:
            splits.append(splits[-1] + desired_length) # splitting into subintervals based on timeframe parameter
        if print_dates:
            for i in range(1, len(splits)):
                print(name + ":" + " "*(4-len(name)), ticker_frame.iloc[splits[i-1]].Date, "to", ticker_frame.iloc[splits[i] - 1].Date)
        for date_split in range(1, len(splits)): # iterating over each subinterval
            start_index, end_index = splits[date_split - 1], splits[date_split]
            data = ticker_frame.iloc[start_index:end_index].reset_index(drop=True)

            # reset
            actions = []
            profits = []
            closed_positions = []
            buy_record = []
            total_reward = 0.

            # initializing prev_sell for reward function
            prev_sell = data.iloc[0].ST_MA

            for idx in range(len(data)-1):

                # extract states and prices
                current_state = data.iloc[idx].state
                current_price = data.iloc[idx].Open
                next_state = data.iloc[idx+1].state
                next_price = data.iloc[idx+1].Open

                action = '' # declare outside following if block
                if validate:
                    action = self.choose_action(current_state, name, epsilon=0) # no random exploration, only exploitation
                else:
                    action = self.choose_action(current_state, name, epsilon=self.epsilon) # some random exploration
                    if next_state not in self.q_table[name]:
                        self.q_table[name][next_state] = [0, 0, 0]
                
                action = 'hold'
                reward = 0 # declare outside following if block
                action_id = ['buy', 'sell', 'hold'].index(action)
                if action=='buy' and self.portfolio['balance'] >= (1+self.commission) * (self.transaction_volume * next_price): # buy
                        # buy at next day's opening price
                        self.portfolio['balance'] -= (1 + self.commission) * (self.transaction_volume * next_price)
                        self.portfolio[name]['holdings'] += self.transaction_volume
                        buy_record.append(next_price)
                        self.buys[name]+=1
                        actions.append('buy')
                        # reward for selling : previous sell price - current buy price
                        reward = prev_sell - next_price
                elif action=='sell' and self.portfolio[name]['holdings']!=0: # sell
                        reward = 0
                        bought_price = buy_record.pop(0)
                        # sell at next day's opening price
                        per_share_profit = (1 - self.commission) * next_price - (1 + self.commission) * bought_price
                        profits.append(self.transaction_volume * per_share_profit)
                        closed_positions.append((bought_price, next_price))
                        self.portfolio[name]['holdings'] -= self.transaction_volume
                        self.portfolio['balance'] += (1 - self.commission) * self.transaction_volume * next_price
                        self.buys[name]-=1
                        # reward for selling : current sell price - bought price
                        reward = next_price - bought_price
                        prev_sell = next_price
                        actions.append('sell')
                else: # hold
                    actions.append('hold')
                    # reward for if you hold : good if price goes up, bad if price goes down
                    reward = self.buys[name]*(next_price - current_price)
                
                total_reward += reward
                if not validate:
                    # temporal update of q-table
                    maximum = max(self.q_table[name][next_state])
                    self.q_table[name][current_state][action_id] = self.q_table[name][current_state][action_id] + \
                        self.alpha * (reward + self.gamma * maximum - self.q_table[name][current_state][action_id])

            # sell remaining holdings
            temp = buy_record.copy()
            for bought_price in buy_record:
                per_share_profit = (1 - self.commission) * next_price - (1 + self.commission) * bought_price
                profits.append(self.transaction_volume * per_share_profit)
                closed_positions.append((bought_price, next_price))
                self.portfolio[name]['holdings']-=self.transaction_volume
                self.portfolio['balance']+=(1-self.commission)*self.transaction_volume*next_price
                self.buys[name]-=1
                # these extra rewards won't affect the q-table but are good for graphing and analysis purposes
                total_reward += next_price - bought_price
                # remove the 'bought prices' of disposed stocks from buy record
                temp.remove(bought_price)

            self.progress_rewards[name].append(total_reward / timeframe)
            self.progress_profits[name].append(sum(profits) / timeframe)

            #================= PRINT SIMULATION STATS ================#
            if print_stats:
                print()
                print('---- Post-simulation portfolio characteristics ----')
                print('Company : {}'.format(name))
                print('Account Balance : {} USD'.format(self.portfolio['balance']))
                print('Holdings : {}'.format(self.portfolio[name]['holdings']))
                print('Next Price : {}'.format(next_price))
                print('Net Present Value : {}'.format(\
                    self.portfolio['balance']+self.portfolio[name]['holdings']*next_price))
                print('Net Profits : {}'.format(sum(profits)))
                print('Net Reward : {}'.format(total_reward))
                print('Normalized Profits : {}'.format(sum(profits) / timeframe))
                print('Normalized Reward : {}'.format(total_reward / timeframe))
                print('Profits:')
                if (len(profits) == 0):
                    print("(none)")
                else:
                    for i in range(len(profits)):
                        print(f"{profits[i]} bought at {closed_positions[i][0]} and sold at {closed_positions[i][1]}")
            #=========================================================#

            #===================== PLOT SIMULATION DATA =====================#
            if graph_stats:
                once_buy = False
                once_sell = False
                temp = data.iloc[:-1].copy()
                temp['action'] = actions
                plt.figure(figsize=(13, 7))
                ax = temp.Open.plot(color='green', label='Price(USD)')
                ax.grid(color='orange', alpha=0.35)
                ax.set_facecolor('white')
                ymin, ymax = ax.get_ylim()
                for idx in range(len(temp)):
                    if temp.iloc[idx].action=='buy':
                        if once_buy:
                            ax.vlines(x=idx, ymin=ymin, ymax=ymax, linestyles='dotted', color='blue', alpha=0.88)
                        else:
                            ax.vlines(x=idx, ymin=ymin, ymax=ymax, linestyles='dotted', color='blue', alpha=0.88, label='buy')
                            once_buy = True
                    elif temp.iloc[idx].action=='sell':
                        if once_sell:
                            ax.vlines(x=idx, ymin=ymin, ymax=ymax, color='red', alpha=0.75)
                        else:
                            ax.vlines(x=idx, ymin=ymin, ymax=ymax, color='red', alpha=0.75, label='sell')
                            once_sell = True            
                plt.xlabel('Simulated Day (#)')
                plt.ylabel('Price in USD')
                plt.title('Trade Simulation Plot : {}'.format(name))
                plt.legend()
                plt.show()
                # input()
            #================================================================#
            self.reset() # reset for next stock


if __name__ == "__main__":
    print("This file contains auxiliary functions for Trader_Bot.py.")
    print("To run the bot, please refer to run.py.")