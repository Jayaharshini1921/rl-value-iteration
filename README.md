# VALUE ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the value iteration algorithm.

## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a gridworld problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. Note that the environment is closed with a fence, so the agent cannot leave the gridworld.

## VALUE ITERATION ALGORITHM
Include the steps involved in the value iteration algorithm
Policy iteration is a method of computing an optimal MDP policy and its value. It begins with an initial guess for the value function, and iteratively updates it towards the optimal value function, according to the Bellman optimality equation. The algorithm is guaranteed to converge to the optimal value function, and in the process of doing so, also converges to the optimal policy.

The algorithm is as follows:

Initialize the value function V(s) arbitrarily for all states s. Repeat until convergence: Initialize aaction-value function Q(s, a) arbitrarily for all states s and actions a. For all the states s and all the action a of every state: Update the action-value function Q(s, a) using the Bellman equation. Take the value function V(s) to be the maximum of Q(s, a) over all actions a. Check if the maximum difference between Old V and new V is less than theta. Where theta is a small positive number that determines the accuracy of estimation. If the maximum difference between Old V and new V is greater than theta, then Update the value function V with the maximum action-value from Q. Go to step 2. The optimal policy can be constructed by taking the argmax of the action-value function Q(s, a) over all actions a. Return the optimal policy and the optimal value function.
## VALUE ITERATION FUNCTION

### Name:Jayaharshini s
### Register Number:212224100024
```
Include the value iteration function
envdesc  = ['FSFF','FFHF','FHFF', 'FGFF']
env = gym.make('FrozenLake-v1',desc=envdesc)
init_state = env.reset()
goal_state = 13 #Enter the Goal state
P = env.env.P


def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward+gamma*V[next_state]*(not done))
        if np.max(np.abs(V-np.max(Q, axis=1))) < theta:
            break
        V = np.max(Q, axis=1)
    pi= lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return V, pi
```
## OUTPUT:
Mention the optimal policy, optimal value function , success rate for the optimal policy.
<img width="692" height="155" alt="Screenshot 2025-09-20 094408" src="https://github.com/user-attachments/assets/f6c2fe53-f134-4da1-b25c-3cd4dc77b2d7" />
<img width="537" height="149" alt="Screenshot 2025-09-20 094420" src="https://github.com/user-attachments/assets/0b1a67af-a133-4559-9365-12c97de841a3" />

<img width="728" height="48" alt="Screenshot 2025-09-20 094428" src="https://github.com/user-attachments/assets/566ffcf7-3678-48be-affc-a89a61fd6a91" />



## RESULT:

Thus, a Python program is developed to find the optimal policy for the given MDP using the value iteration algorithm.
