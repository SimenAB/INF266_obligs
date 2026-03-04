## Part 1
#### Exercise 1: Monte Carlo methods
#### 1.1 Consider a square with a side length of 1. Define analytically the ratio X between the area of the circle inscribed in the square and the square itself.

Square side length = 1, so the area of the square is also 1. The inscribed circle has diameter 1, so the radius is $1/2$.
Circle area:
$$A∘=πr^{2}=π(\frac{1}{2})=\frac{π}{4}$$
So the ratio X will be:
$$X=\frac{A∘}{A□​}=\frac{\frac{π}{4}}{1}=\frac{π}{4}$$
#### 1.2 Equate the analytical ratio X that you have computed to an empirical ratio ˆX you will compute via Monte Carlo, and isolate π.

$$X_{hat}≈X=\frac{π}{4}$$

$$π=4X_{hat}$$
#### 1.3 Sample 10000 pairs of numbers (x1, x2) generated as follows: x1 ∼ Unif[0, 1] are i.i.d. samples from a uniform distribution over [0, 1]; x2 ∼ Unif[0, 1] are i.i.d. samples from a uniform distribution over [0, 1].

rng = np.random.default_rng(seed)
x1 = rng.uniform(0.0, 1.0, size=N)
x2 = rng.uniform(0.0, 1.0, size=N)

#### 1.4 Estimate ˆX from the samples, by taking the ratio of points falling within and without the circle.

N = 10000,
seed = 67
X_hat = 0.779000

#### 1.5 Estimate π.

N = 10000
seed = 67
π = 3.116000

#### 1.6 Generate a new dataset of 10000 pairs of numbers (x1, x2) generated as follows: x1 ∼ Unif[0, 1] are i.i.d. samples from a uniform distribution over [0, 1]; x2 = x1 + Unif[−0.1, 0.1], that is, x2 is equal to x1 plus a i.i.d. random noise sampled from a uniform distribution over [−0.1, 0.1].

rng = np.random.default_rng(seed)
x1 = rng.uniform(0.0, 1.0, size=N)
eps = rng.uniform(-noise_width, noise_width, size=N)
x2 = x1 + eps
out_of_bounds = np.mean((x2 < 0.0) | (x2 > 1.0))
if clip_x2:
	x2 = np.clip(x2, 0.0, 1.0)

#### 1.7 Estimate the new ˆX and π

X_hat = 0.698400
pi_hat = 2.793600
Out of bounds fraction of x2 (x2<0 or x2>1): 5.19%

#### 1.8 How do the two estimates change and why?

From my runs:
- Case 1
	$X_{hat}=0.7790$ and $π=3.1160$

- Case 2
	$X_{hat}= 0.6984$ and $π=2.7936$

So the estimates decrease in Case 2:
By 0.0806 for X_hat, and 0.3224 for π.

#### Why do they change?
The relation $X_{hat}≈\frac{π}{4}$ is valid only when the points are sampled uniformly over the unit square
In that setting,
$$X=P(InsideCircle)=\frac{Area(Circle)}{Area(Square)}=\frac{π}{4}$$

- Case 1 samples (x_1,x_2) uniformly on [0,1]^2 (since (x_1) and (x_2) are i.i.d. uniform). Therefore X_hat is an estimate of the area ratio π/4, and 4*X_hat estimates π.
- Case 2 does not sample uniformly on the square because (x_2 = x_1 + variable) makes the points strongly dependent and concentrated near the diagonal (x_2 ≈ x_1). This changes the probability of landing inside the circle, so X_hat is no longer estimating the area ratio π/4.
Additionally, about 5.19% of samples have (x_2 < 0) or (x_2 > 1). These points are automatically outside the circle (which lies entirely inside the square), further reducing X_hat.

#### Exercise 2: MC and MABs

#### 2.1 What sort of problem are we solving in a MAB: prediction or control?

In a multi armed bandit we are solving a control problem, not a prediction problem.

#### 2.2 In a MAB you compute expected rewards of arms. Would this correspond to a state-value function or an action-value function?

It corresponds to an action value function. In a MAB there is one state that is reset everytime, and each arm is an action, we calculate the expected value of each arm or the value of each action.

#### 2.3 What would correspond to a trajectory in a MAB? What would be a collection of trajectories?

In a MAB a trajectory is just the sequence of (chosen arm/action, observed reward) pairs over time, because there’s only one state and each interaction is one step. A collection of such trajectories is called episodes. In bandits, each episode is often considered one step (pull -> reward -> reset), but if you view a run over a horizon T, that whole run is also a trajectory.

#### 2.4 To what MC family of algorithms would you relate the MAB algorithms we studied: MC prediction or MC control?

They line up with MC control family of algorithms. Because the bandit algorithms are not just evaluating a fixed policy, they learn or improve the policy while estimating action values/preferences to maximize reward.

#### 2.5 What would correspond to MC prediction in a MAB?

In a multi-armed bandit (MAB), MC prediction corresponds to policy evaluation: the arm-selection policy is fixed, and we only estimate the expected reward of each arm from sampled experience (no policy improvement).
Let a fixed policy be π(a) (e.g., always pull arm 2, or choose arms uniformly at random).
Generate experience as a sequence of action–reward pairs:
(a_1, r_1), (a_2, r_2)
Then estimate the action-values under that policy:
$q_\pi(a) = \mathbb{E}[R\mid A=a]$
A Monte carlo estimate is the sample average of observed rewards for each arm:
$\hat q(a)\leftarrow\frac{1}{N(a)}\sum_{i: a_i=a}r_i$

This is MC prediction because the policy π is not updated; we only predict or evaluate the expected rewards (action values) based on sampled returns.

#### 2.6 What is the purpose of ϵ in MABs and in MC control?

In MAB ϵ sets the exploration rate. With probability ϵ
ϵ you explore (pick a random arm), and with probability 1−ϵ you exploit (pick the currently best estimated arm).
In MC control ϵ makes the policy ϵ soft, meaning every action has non zero probability of being selected.

#### 2.7 Could we reduce a standard RL problem solved by MC to a MAB problem? If so, what would we lose?

Yes we can reduce a standard RL problem to something bandit like, but only by throwing away the sequential part. 
What would be lost:
- State dependence / context: In RL, the right action depends on the current state s. A plain MAB can’t represent same action, different outcomes depending on s.
- Transitions and long-term consequences: RL actions change what state you’ll be in next. In a MAB, actions don’t affect future observations.
- Delayed reward + credit assignment: MC in RL learns from returns (sum of rewards over time). In a MAB you only model immediate reward per arm, so you lose learning about actions whose benefit shows up later.
#### Exercise 3: Importance sampling

#### 3.4 Compute the ratio ρ between the distribution probability of R and S

[1.16666667 0.85714286 0.83333333 2.0]

#### 3.5 Suppose now that you only have the samples from S that you have already collected. Suppose, also, that someone let you know the value of ρ. Use importance sampling to compute the empirical expectation of ˆE[R] from the samples of S and ρ.

E_hat[R] = 6.023452380952381

#### 3.8 How do the results you computed differ? What explains the differences?

The differences come from how similar the behavior policy is to R: when
$p_{\text{behavior}}$ is close to (p_R), importance weights stay near 1 and importance sampling has low variance (stable estimates). When the behavior
distribution assigns tiny probability to outcomes that R cares about (like 0 and 20 in S''), the weights explode, causing high variance importance sampling estimates that can fluctuate a lot across runs even if one run looks accurate.

#### Exercise 4: Q-Learning

#### 4.1 Given that we do off-policy control, why do we not need to rely on importance ratio in Q-learning?

In off policy learning, importance sampling is needed when you estimate an expected update under a target policy pi, but your data come from a different behavior policy beta.
- Then you must correct using the ratio:
$$\frac{\pi(a|s)}{\beta(a|s)}$$

Q learning is different because its target is not an expectation under pi.
Instead, it uses the max over actions (the greedy/optimal choice): $$Q(s,a)\leftarrow Q(s,a)+\alpha\Big(r+\gamma \max_{a'} Q(s',a')-Q(s,a)\Big)$$
Since the update uses $max_{a'} Q(s',a')$, it does not depend on action probabilities like $pi(a'|s')$.
Therefore there is no probability mismatch to correct, and no importance ratio is needed.