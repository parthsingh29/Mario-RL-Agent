# Mario-RL-Agent

## Theory
Let’s say we want to design an algorithm that will be able to complete the first level of the Super Mario Bros. game. How will we do that?

The goal of this game is quite simple — get the flag at the rightmost end of the level as fast as possible. To do that, we need to observe the current state of the game s, move Mario by pressing control buttons (let’s call it an **action a**), and check how far we are from the goal after that (in reinforcement learning action’s performance is evaluated by **reward r**), observe some new **state s’** (new frame in our case) and make the next move. As a result, we will get a sequence of actions, states and rewards called a trajectory:

![image](https://user-images.githubusercontent.com/77194307/176742857-071031c7-237f-4008-80bf-fe8d0324d584.png)
---

In more formal terms, this environment can be described as a Markov Decision Process if we act by considering multiple frames at once to incorporate the character’s speed. It means that transitions between states only depend on the latest pair of state and action, without a prior history.

Let’s call an estimator (it could be a neural network) that takes a state and tells us which action to take a policy, so our goal is to maximize the expected return over a trajectory

![image](https://user-images.githubusercontent.com/77194307/176743103-247c5350-3ee6-4751-a95e-d7ce8079d587.png)

## Double Deep Q-Network
Another approach to solving reinforcement learning problems is to approximate an optimal action-value function, which gives us the maximum expected return if we start in the state s and take some action a

![image](https://user-images.githubusercontent.com/77194307/176743352-eb0fb856-b61c-4411-8bcc-ef52b22e32c0.png)
---
This function obeys the Bellman equation, which tells us that if the optimal value Q for the state s is known for all possible actions a, then the optimal strategy is to select the action a maximizing the reward plus the value of state you land next

![image](https://user-images.githubusercontent.com/77194307/176743459-41d0d1dc-0d13-49b4-ab10-23a2274b4d4a.png)
---
The simplest (and the oldest) algorithm for estimation optimal Q values for all state-action pairs is Q-learning. It assumes we have a large look-up table for all such pairs, and within one episode, estimation occurs as follows:

* Observe the current state s at time step t.
* Select and perform an action a with the highest Q value for the current state (or pick the random action sometimes — it’s called the ϵ-greedy approach).
* Observe the next state s’ and the reward r’ at time step t+1.
* Update the Q-values according to the formula below. Here we estimate Q’ out of the best Q values for the next state, but which action a’ leads to this maximal Q is not quite important

![image](https://user-images.githubusercontent.com/77194307/176743814-341fc0b8-cc14-4218-a244-19da4be9ea1e.png)
---
Memorization of all the Q values for all state-action pairs is impractical. A much better way is to approximate Q values using a function. This is where the neural network comes into play.

An algorithm called Deep Q-Network greatly improved training stability and reduced resource requirements by introducing two innovative approaches — experience replay and the periodically updated target network.

Experience replay mechanism uses a single replay memory of fixed size where the N last (s, a, r, s’) tuples are stored. These samples are randomly drawn from the replay memory during training. It significantly increases sample efficiency and reduces correlations between sequences of observations.

A periodically updated target network means keeping a separate cloned instance of the neural network whose weights are frozen and synced with the leading network only every C step. This network is used to estimate target Q values during training. This mechanism reduces the impact of short-term fluctuations and thus stabilizes the training process.

The loss function for the Deep Q-Network looks like this (where U(D) is a uniform random sampling from the replay memory)

![image](https://user-images.githubusercontent.com/77194307/176743936-83d0a944-9792-47c3-a94c-13f5bf79c1f4.png)
---
Unfortunately, Q-Learning (and the DQN based on it) algorithm has a major drawback — it tends to significantly overestimate action values. This happens because the Q-Learning algorithm uses the same set of samples to find the best action (with the highest expected reward) and to estimate the action value. So if the action’s value was overestimated and it was chosen as the best action, then the Q value is overestimated too. If the overestimations over all the Q values are not uniform (this largely depends on the environment’s transition rules and size of the action space), we will spend more time exploring such non-optimal states, and the learning process will be slow. You can find a more formal explanation here.

That’s why in 2010, Hado van Hasselt introduced a new way to estimate Q values called Double Q-learning. This method uses two estimators A and B, which are updated alternately. If we want to update estimator A, then the Q value for the next step is evaluated using estimator B. This approach fixes the overestimation problem because one of the estimators might see samples that overestimate action a1, while the other sees samples that overestimate action a2.

So in 2015, the same team published the paper with an updated version of the DQN algorithm called Double Deep Q-Network that has the following loss function

 ![image](https://user-images.githubusercontent.com/77194307/176744049-9fabf81f-aa35-4ba5-b798-7d7db7d58bae.png)
---
As you can see here, the selection of the action in the argmax is decided by the online network, so this value estimation follows the greedy policy according to the current values, but we use the target network to evaluate the value of this policy. We can update the target network by periodically syncing its weights with the online network or switching roles of these two networks.



## Practice
In the practice section of this article, we will use the first level from the Super Mario Bros. game as an environment. By default, the single observation is a 240 x 256 pixels RGB image, so we need to write a few wrappers to transform it to a grayscale image with a resolution of 84 x 84 pixels. Also, not all observations are useful, so that we will use only every fourth observation and stack them together.
Now we can create our environment and set the random seeds to the fixed values to get reproducible results. Also, for simplicity, the action space was limited to two actions — moving to the right and a combination of moving to the right and jumping.

![image](https://user-images.githubusercontent.com/77194307/176744433-1aa5cad4-ccd4-4cfa-b337-928f2ffc08f8.png)
