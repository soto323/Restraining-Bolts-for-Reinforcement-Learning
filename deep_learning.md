# Deep Learning: problems and possible solutions



## Environment Specific problem:

* Reward sparsisity
* Partial Observability 
  

## DQN and DDQN:

* We didn't address partial observability .
* Using random sampling of batches $\rightarrow$ it may sample something inappropriate for training. Hence we are never sure if it's right or not. Solution: Use prioritised experience replay.
* Using different optimizers and loss functions.


## Actor Critic (A2C):

* We used the same network to predict the value and action. Instead we may use different networks 
  * 1. Actor
    2. Critic
* Possibly use asynchronous A3C https://www.geeksforgeeks.org/asynchronous-advantage-actor-critic-a3c-algorithm/

## Global possible solutions

* All the above metnioned networks are relatively small, so we can perform cheap, hyperparater tuning (in terms of computational complexity), i.e.

  * Population based training: Training multiple networks simultaneously, and let the best ones continue training for longer, approx 1M episodes.
  * Implementation from another course project:  https://github.com/Ostyk/population-based-training-of-NNs

  























