Goal of this exercise is to teach an agent to collect bannanas. To do that we use a technique called "reinforcement learning" or RL for short.

# Learning algorithm

Learning algorithm that we will use is called "Deep Q Learning" (DQN). The algorithm can be divided into tvo parts - sampling and learning. It works as follows:

Sampling:
1. Choose action A given state S using policy P.
2. Take action A, observe reward R and next state S'.
3. Store experience tuple (S, A, R, S') into replay memory M.

Learning:
4. Obtain random batch of tuples B from M.
5. Set target y = r + discount_rate * max_a(q(S', a, w')
6. Update Q network weights: delta_w = learning_rate * (y - q(S, a, w)) * grad_w(q(S, a, w))
7. Every C steps set w' = w

max_a(f(a)) - function that returns maximum value of f(a) with respect to a. 

grad_w(f(w)) - function that returns gradient of f(w) with respect to w.

The algorithm tries to maximise expected cumulative reward.


# Model architecture

DQN used for this exercise is defined in `model.py` file.


# Code walkthrough

The 