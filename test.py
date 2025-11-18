# plot the training progress of the rewards.csv
import pandas as pd
import matplotlib.pyplot as plt

# read the rewards.csv
rewards = pd.read_csv('ppo_model_logs/rewards.csv')

# plot the rewards
plt.plot(rewards['reward'])
plt.show()
