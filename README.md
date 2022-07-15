# BA_Deep_Reinforcement_Learning

- [ ] Introduction here

## Quick Start

- [ ] Installation steps & tested versions
- [ ] Give different possibilites of using this repo:
    - [ ] Explain how to use the trained models
    - [ ] Explain how to follow the exact steps I made 

## Experiment Structure

- [ ] Explain how I structured my experiments and describe them

The experiments can be broken down into two main phases.
**Phase 1** consists of all experiments where the RiskyPathEnv is
used with the `TensorObsWrapper`. When the wrapper's `step()` method is called
by the SB3-algorithms (I use DQN and A2C), the returned observation is a
relatively low-dimensional input when compared to rgb-arrays that are
returned in the `ImgObsWrapper`. The effect of different environment
specifications on the learning algorithms' performance can thus be
analyzed in a more efficient way, as learning on this kind of input drastically
reduces complexity and computation time.
**Phase 2** consists of all experiments where the observation is an rgb-array
(returned by the `ImgObsWrapper`). For learning on this kind of input, the
sb3-algorithms will use `CnnPolicy` TODO

Below you can find a listing of all experiments, annotated with the environment
settings and the learning & network parameters of the algorithm employed.
Furthermore, the path to the saved model (if available) and the tensorboard
logs are provided.