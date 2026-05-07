# Part 1 - Multi Armed Bandits

### 1. How would this modified problem fit in the MAB or in the MDP formalism? What challenges arise in the current setting?

This is now considered a non-stationary MAB problem (restless bandit). The difference from the stationary MAB is that there is no single optimal arm, the 'best' medicine is a moving target. The main challenge is that historical data becomes outdated, creating a 'memory' problem. Consequently, the agent can never stop exploring, as it must constantly validate if the environment has shifted to ensure the learned policy remains effective.

The primary challenge in this setting is that the environment is non-stationary, which leads to the following specific problems:

- In a standard MAB, more data leads to more certainty. Here, historical data becomes a liability. If the agent relies on observations from the beginning of the trial, it may be misled by "stale" averages that no longer reflect the current effectiveness of the medicines.

- In stationary problems, the agent eventually shifts almost entirely to exploitation. In this non-stationary case, exploration can never end. The agent must continuously sample sub-optimal drugs to detect if their effectiveness has improved or if the current "best" drug has degraded.

- There is a difficult trade-off between reacting quickly to a change in medicine effectiveness and not being misled by random noise in a single patient's reaction.

- The agent must distinguish between gradual assuefaction and sudden shocks, as each requires a different strategy for updating the internal model.

### 2. Assume your aim is still to minimize regret or maximize reward. How would you change or adapt a MAB algorithm to deal with this scenario in which you do not know when changes in effectiveness happen? Explain and justify your algorithmic choices. Describe assumptions and/or dependence of your solution on the parameters of the problem.

To adapt the MAB algorithm for medicines whose effectiveness changes over time, we must shift away from the full memory of standard models. In a stationary setting, an algorithm like UCB (upper confidence bound) treats every data point from the past as equally valid. To solve this, we implement a sliding window UCB or a discounted UCB approach, both of which operate on the principle of forgetting after some time.

In the sliding window approach, the algorithm is modified to calculate the empirical mean and the exploration bonus using only the most recent N observations. By effectively "deleting" the old data, we prevent the algorithm from being anchored to an outdated reality where a drug might have been more effective than it is now. Alternatively, a discounted UCB uses a decay factor to exponentially decrease the weight of past observations. This ensures that the agent's current strategy is always dictated by the most recent biological responses of the population, allowing the confidence intervals to remain wide enough that the agent never stops exploring the other arms.

The primary justification for this change is the elimination of historical bias. In the original UCB, as the number of trials increases, the confidence intervals shrink, and the algorithm becomes more "stubborn" in its choices. By limiting the data to a recent window we ensure the algorithm remains agile. If a medicine’s effectiveness suddenly drops, the fewer samples being considered allow the calculated average to plummet quickly, triggering the agent to switch to a more effective alternative.

The success of this solution depends heavily on the tuning of its parameters, specifically the window size or the discount factor. These parameters represent an assumption about the volatility of the environment, we must choose a window small enough to react to changes in effectiveness, but large enough to filter out the random noise of individual patient reactions. We also assume that the "random time steps" at which the drugs change are not so frequent that the algorithm is unable to gather enough recent data to make an informed decision. If the effectiveness shifts faster than the agent can sample, the uncertainty remains too high to effectively minimize the regret.

### 3. Download the file bandit.py and run the environment Bandits final() representing the scenario discussed above for (at least) 1000 episodes. Run the algorithmic solution proposed above alongside (at least) one of the standard MAB algorithm discussed in the course. Compare and comment on the result.

To test the proposed solution, we ran the provided Bandits_final() environment for 5000 episodes, where one episode corresponds to one arm pull. The environment has three arms, rewards are sampled from normal distributions, and the average effectiveness of the drugs changes randomly according to the implementation in bandit.py. This makes the problem non-stationary because an arm that was previously optimal can suddenly become sub-optimal.

We compared a standard UCB algorithm with the sliding-window UCB proposed above. The standard UCB keeps all observations from the beginning of the run, while the sliding-window UCB only estimates arm values from the most recent 100 observations. Both algorithms use the same UCB action rule, but the sliding-window version computes both the empirical mean and the uncertainty term from recent data only:

```text
score_i = mean_i(recent window) + c * sqrt(log(window size) / pulls_i(recent window))
```

For reference, we also ran an epsilon-greedy baseline with epsilon = 0.1. Regret was measured against the best arm according to the current means in the environment.

| Algorithm | Total reward, one run | Regret, one run | Optimal action rate, one run |
| --- | ---: | ---: | ---: |
| Standard UCB | 17092.87 | 3156.00 | 0.634 |
| Sliding-window UCB, window = 100 | 19499.87 | 749.00 | 0.876 |
| Epsilon-greedy, epsilon = 0.1 | 11940.45 | 5610.00 | 0.140 |

Since a single run can depend heavily on when the environment changes, we also averaged over 100 independent runs of 5000 episodes each, using seeds 0 to 99:

| Algorithm | Average total reward | Average regret | Average optimal action rate |
| --- | ---: | ---: | ---: |
| Standard UCB | 16345.96 +- 2226.64 | 3605.59 +- 1585.57 | 0.579 +- 0.149 |
| Sliding-window UCB, window = 100 | 19005.08 +- 1686.81 | 946.47 +- 135.47 | 0.788 +- 0.056 |
| Epsilon-greedy, epsilon = 0.1 | 14882.58 +- 2216.43 | 5161.25 +- 1707.65 | 0.465 +- 0.145 |

The sliding-window UCB performed best in these experiments. It achieved the highest average reward and the lowest average regret, which supports the argument from Exercise 1.2: in a non-stationary bandit, old observations should not be trusted forever. Standard UCB still performs reasonably well, but it is slower to recover after a change because its estimates are anchored by all previous samples. This is visible in the higher regret and larger variance across runs.

The epsilon-greedy baseline was worse than both UCB methods. Its fixed exploration probability guarantees that it keeps trying other arms, which is useful in a changing environment, but it does not direct exploration toward uncertain arms as efficiently as UCB. It therefore wastes more pulls on clearly weak arms.

The result depends on the window size. A window of 100 works well here because the environment only checks for possible changes every 200 episodes, so the algorithm has enough recent samples to estimate the current means while still forgetting old regimes quickly. A much larger window would behave more like standard UCB and adapt too slowly, while a much smaller window would react quickly but become too sensitive to reward noise.
