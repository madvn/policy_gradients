import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# checking to see if the plots need to be shown
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--visualize', action="store_true")
args = parser.parse_args()

def probs_from_prefs(prefs):
    ''' probability from preferences '''
    return np.exp(prefs)/(np.sum(np.exp(prefs)))

def running_average(average, new_item, count):
    ''' add item to running average '''
    return ((average*count)+new_item)/(count+1) #np.mean([average,new_item])

# config
num_arms = 4
alpha = 0.1
baseline = running_average
savefig_path = './'
savefig_names = ['baseline_avg.png', 'true_returns.png']

class MultiArmBandits:
    ''' Class to init and play stationary MultiArmBandits'''
    def __init__(self, num_arms):
        self.arms = set(np.arange(num_arms))
        self.arm_reward_means = [0, 0.5, -0.5, 1]
        self.arm_reward_stds = np.random.rand(num_arms)

    def play(self, selected_arm):
        ''' return sampled reward for selected arm'''
        return np.random.normal(loc=self.arm_reward_means[selected_arm], scale=self.arm_reward_stds[selected_arm])

# initial values for arm preferences and other vars
arms = np.arange(num_arms)
arm_preferences = np.ones(num_arms)/num_arms
arm_probs = probs_from_prefs(arm_preferences)
baseline_value = 0.
baseline_values = [baseline_value]
arm_choices = []
time = 0

# creating the bandits
bandits = MultiArmBandits(num_arms)

# playing and learning
while time<1000:
    # take action and get reward
    selected_arm = np.random.choice(arms, p=arm_probs)
    selection_mask = np.zeros(num_arms)
    selection_mask[selected_arm] = 1
    reward_t = bandits.play(selected_arm)

    # learn
    arm_preferences += alpha*(reward_t - baseline_value)*(selection_mask-arm_preferences[selected_arm])
    baseline_value = baseline(baseline_value, reward_t, time)
    arm_probs = probs_from_prefs(arm_preferences)

    # data collection
    baseline_values.append(baseline_value)
    arm_choices.append(selected_arm)
    time+=1

plt.figure(1)
plt.plot(baseline_values)
plt.xlabel('time')
plt.ylabel('baseline avg. reward')
plt.savefig(os.path.join(savefig_path,savefig_names[0]))

plt.figure(2)
choice_hist, _ = np.histogram(arm_choices, bins=num_arms)
_ = [plt.bar(armi, choice_histi) for armi,choice_histi in zip(arms, choice_hist)]
plt.xlabel('arm_choice')
plt.ylabel('frequency of arm being chosen')
plt.legend(['Arm return mean = {}'.format(arm_reward) for arm_reward in bandits.arm_reward_means])
plt.savefig(os.path.join(savefig_path,savefig_names[1]))

print('The payoffs from the arms were as follows:')
print('means: ', bandits.arm_reward_means)
print('std: ', bandits.arm_reward_stds)

if args.visualize:
    plt.show()
