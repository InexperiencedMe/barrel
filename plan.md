### Vision:

Barrel raises the bar of reinforcement learning. It's a Deep RL library interfacing with Unity engine, that is used for AI research to experiment and explore different pathways. It's an AI playground interface.

Not speed competitive yet. We prioritize features and extending the current possibilities.


### Further main goals:

1. Finish SAC implementation
- Make it easy to change the sizes and number of hidden layers (save them in checkpoint)
- Can we still make it enclosed in a class? Didnt work last time but we can try again after many changes

2. Improve the API - make parts of the code reusable
- Make universal graphing function
- Saving, loading checkpoint from a function
- Initialization of buffers from a function
- Managing buffers with a function (adding memories, resetting buffers, adding reward, appending reward)

3. More algorithms!
- PPO
- Genetic algorithm
- Curiosity-Driven agent
- MuZero