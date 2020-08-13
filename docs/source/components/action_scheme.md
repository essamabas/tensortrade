# Action Scheme

An `ActionScheme` defines the action space of the environment and interprets the
action of an agent and how it gets applied to the environment.

For example, if we were using a discrete action space of 3 actions (0 = hold, 1 = buy 100 %, 2 = sell 100%), our learning agent does not need to know that returning an action of 1 is equivalent to buying an instrument. Rather, the agent needs to know the reward for returning an action of 1 in specific circumstances, and can leave the implementation details of converting actions to trades to the `ActionScheme`.

Each action scheme has a `perform` method, which will interpret the agent's specified action into a change in the environmental state. It is often necessary to store additional state within the scheme, for example to keep track of the currently traded position. This state should be reset each time the action scheme's reset method is called, which is done automatically when the parent `TradingEnv` is reset.

### What is an Action?

This is a review of what was mentioned inside of the overview section and explains how a RL operates. You'll better understand what an action is in context of an observation space and reward. At the same time, hopefully this will be a proper refresher.

An action is a predefined value of how the machine should move inside of the world. To better summarize, its a _command that a player would give inside of a video game in respose to a stimuli_. The commands usually come in the form of an `action_space`. An `action_space` defines the rules for how a user is allowed to act inside of an environment. While it might not be easily interpretable by humans, it can easily be interpreted by a machine.

Let's look at a good example. Let's say we're trying to balance a cart with a pole on it (cartpole). We can choose to move the cart left and right. This is a `Discrete(2)` action space.

- 0 - Push cart to the left
- 1 - Push cart to the right

When we get the action from the agent, the environment will see that number instead of a name.

![Watch Link Run Around In Circles](../_static/images/cartpole.gif)

An `ActionScheme` supports any type of action space that subclasses `Space` from `gym`. For example, here is an implementation of an action space that represents a probability simplex.

```python
import numpy as np

from gym.spaces import Space

class Simplex(Space):

    def __init__(self, k: int) -> None:
        assert k >= 2
        super().__init__(shape=(k, ), dtype=np.float32)
        self.k = k

    def sample(self) -> float:
        return np.random.dirichlet(alpha=self.k*[3*np.random.random()])

    def contains(self, x) -> bool:
        if len(x) != self.k:
            return False
        if sum(x) != 1.0:
            return False
        return True
```



## Default
The default TensorTrade action scheme is made to be compatible with the built-in order management system (OMS). The OMS is a system that is able to have orders be submitted to it for particular financial instruments.

### Simple

<br> **Overview** <br>
Blank.
<br> **Action Space** <br>
Blank.
<br> **Perform** <br>
Blank.
<br> **Compatibility** <br>
Blank.
<hr>


### ManagedRisk

<br> **Overview** <br>
Blank.
<br> **Action Space** <br>
Blank.
<br> **Perform** <br>
Blank.
<br> **Compatibility** <br>
Blank.
<hr>

### BSH

<br> **Overview** <br>
The buy/sell/hold (BSH) action scheme was made to capture the simplest type of
action space that can be made. If the agent is in state 0, then all of its net worth
is located in our `cash` wallet (e.g. USD). If the agent is in state 1, then all of
its net worth is located in our `asset` wallet (e.g. BTC).

<br> **Action Space** <br>
* `Discrete(2)`

<br> **Perform** <br>
Below is a table that shows the mapping `(state, action) -> (state)`. <br>

State | Action | Meaning |
----- | ------ | ------- |
0 | 0 | Keep net worth in `cash` wallet (HOLD) |
0 | 1 | Transition net worth from `cash` wallet to `asset` wallet (BUY) |
1 | 0 | Transition net worth from `asset` wallet to `cash` wallet. (SELL) |
1 | 1 | Keep net worth in `asset` wallet (HOLD) |

<br> **Compatibility** <br>
* `PBR` (position-based reward)
<hr>

### Pairs
Blank.
<br> **Overview** <br>
Blank.
<br> **Action Space** <br>
Blank.
<br> **Perform** <br>
Blank.
<br> **Compatibility** <br>
Blank.
<hr>