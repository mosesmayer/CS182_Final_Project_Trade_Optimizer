from Trader_Bot import Stockinfo, Trader
import matplotlib.pyplot as plt
import numpy as np

stocklist = ["JNJ", "MSFT", "TSM", "WMT", "PG", "NVDA", "DIS", "MMM", "AMD"]

stocks = []
for i in stocklist:
    s = Stockinfo(i, str("data_"+i+".csv"))
    stocks.append(s)

# state_model can be MACD, ST_tau, LT_tau, Stochastic
state_model_ = "MACD"
assert state_model_ in ["MACD", "ST_tau", "LT_tau", "Stochastic"], "Invalid state model selected."
t = Trader(stocks = stocks, epsilon = 0.1, alpha = 0.1, gamma = 0.9, state_model = state_model_)

timeframes = [1.0, 0.5, 0.25, 0.1] # list of timeframes to train on
episodes = 100 # episodes per timeframe (timeframes < 1.0 will perform multiple sub-episodes within each episode)
cumulative_eps = list(np.cumsum([int(episodes / i) for i in timeframes]))
num_episodes = cumulative_eps[-1] # total number of training episodes


# begin training
current_ep = 1
for tf_idx in range(len(timeframes)):
    for _ in range(episodes):
        tf = timeframes[tf_idx]
        eps = int(1/tf)
        ep_str = str(current_ep)
        if eps != 1:
            ep_str += "-" + str(current_ep + eps - 1)
        print("\n\n===========TRAIN "+ep_str+"===========")
        current_ep += eps
        t.simulate_trader(df = t.train_set, timeframe = tf, print_dates=False, print_stats=False, graph_stats=False)

eval_tframes = [0.25, 0.5] # list of timeframes to evaluate on
eps_validate = 30 # episodes per timeframe (timeframes will perform multiple sub-episodes within each episode)
cumulative_eps_val = list(np.cumsum([int(eps_validate / i) for i in reversed(eval_tframes)]))
num_ep_validate = cumulative_eps_val[-1] # total number of testing episodes


# begin evaluation
print("\n\n===========EVALUATE===========")
for tf_idx in range(len(eval_tframes)):
    for i in range(eps_validate):
        t.simulate_trader(df = t.test_set, timeframe = eval_tframes[tf_idx], validate=True, print_dates=False, print_stats=False)

# print evaluation statistics
for stock in stocklist:
    print("\n"+stock+":")
    all_rewards = t.progress_rewards[stock]
    all_profits = t.progress_profits[stock]
    print("  Average normalized reward:")
    print("    Total:", sum(all_rewards[len(all_rewards)-num_ep_validate:])/num_ep_validate)
    print("    0.25 timeframe:", sum(all_rewards[len(all_rewards)-cumulative_eps_val[1]:len(all_rewards)-cumulative_eps_val[0]])/eps_validate)
    print("    0.5 timeframe:", sum(all_rewards[len(all_rewards)-cumulative_eps_val[0]:])/eps_validate)
    print("  Average normalized profit:")
    print("    Total:", sum(all_profits[len(all_profits)-num_ep_validate:])/num_ep_validate)
    print("    0.25 timeframe:", sum(all_profits[len(all_profits)-cumulative_eps_val[1]:len(all_profits)-cumulative_eps_val[0]])/eps_validate)
    print("    0.5 timeframe:", sum(all_profits[len(all_profits)-cumulative_eps_val[0]:])/eps_validate)


# graphing reward and profit over the episodes

print("\n\nReward slopes:")
plt.figure(1)
for stock in stocklist:
    all_rewards = t.progress_rewards[stock]
    graph_rewards = np.asarray(all_rewards[:-num_ep_validate])
    x = np.arange(graph_rewards.size)

    m,b = np.polyfit(x, graph_rewards, 1)

    print(stock + ":", m)
    plt.plot(x, graph_rewards)
    plt.title("Normalized rewards (raw) over " + str(num_episodes) + " episodes using " + state_model_)
plt.legend(stocklist)
for i in cumulative_eps[:-1]:
    plt.axvline(i, color='k', linestyle='dotted')

plt.show() # raw rewards


plt.figure(2)
for stock in stocklist:
    all_rewards = t.progress_rewards[stock]
    graph_rewards = np.asarray(all_rewards[:-num_ep_validate])
    x = np.arange(graph_rewards.size)

    m,b = np.polyfit(x, graph_rewards, 1)

    plt.plot(x, m*x+b)
    plt.title("Normalized rewards (best fit lines) over " + str(num_episodes) + " episodes using " + state_model_)
plt.legend(stocklist)

plt.show() # reward best fit lines


print("\nProfit slopes:")
plt.figure(3)
for stock in stocklist:
    all_profits = t.progress_profits[stock]
    graph_profits = np.asarray(all_profits[:-num_ep_validate])
    x = np.arange(graph_profits.size)

    m,b = np.polyfit(x, graph_profits, 1)

    print(stock + ":", m)
    plt.plot(x, graph_profits)
    plt.title("Normalized profits (raw) over " + str(num_episodes) + " episodes using " + state_model_)
plt.legend(stocklist)
for i in cumulative_eps[:-1]:
    plt.axvline(i, color='k', linestyle='dotted')

plt.show() # raw profits


plt.figure(4)
for stock in stocklist:
    all_profits = t.progress_profits[stock]
    graph_profits = np.asarray(all_profits[:-num_ep_validate])
    x = np.arange(graph_profits.size)

    m,b = np.polyfit(x, graph_profits, 1)

    plt.plot(x, m*x+b)
    plt.title("Normalized profits (best fit lines) over " + str(num_episodes) + " episodes using " + state_model_)
plt.legend(stocklist)

plt.show() # profit best fit lines


# display a few buy/sell graphs using the testing data (another means of evaluation)
print("\n\n===========GRAPHING===========")
for tf_idx in range(len(timeframes)):
    t.simulate_trader(df = t.test_set, timeframe = timeframes[tf_idx], validate=True, print_dates=False, print_stats=False, graph_stats=True)