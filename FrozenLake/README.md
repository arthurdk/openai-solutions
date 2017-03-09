# Solution to [FrozenLake-v0](https://gym.openai.com/envs/FrozenLake-v0)

This is a Tensorflow implementation of a Q-Learning method using Neural Networks.

## Results

By using a Q-Learning method with Neural Networks I solved the problem in less than 100 episodes.

You can find my best [evaluation directly on Gym](https://gym.openai.com/evaluations/eval_5LmYbJtQOmyyTwi8p8TA).

## Reproduce

Simply launch the following command:

```
python frozen_lake.py --stats
```
## References

This solution is heavily based on this blog post:

* [Simple Reinforcement Learning with Tensorflow Part 0: Q-Learning with Tables and Neural Networks](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.kxecl0tug) by Arthur Juliani

and also on this explanation of **Reinforcement Learning**:

* [Reinforcement Learning and DQN, learning to play from pixels](https://rubenfiszel.github.io/posts/rl4j/2016-08-24-Reinforcement-Learning-and-DQN.html) by Ruben Fiszel
