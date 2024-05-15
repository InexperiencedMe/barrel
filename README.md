# Barrel (BaRL)
## Raising the (lowest) bar of RL implementations

Development phase. Discrete action space doesn't work, continuous action space works just fine, but we haven't tested for more than 20k steps


### What is it?

It's a beginning of a library that conveniently interfaces with Unity engine to run and test RL algorithms on more complex custom environments. My dream is a model zoo, where I'd construct a beautiful video game scene to showcase RL algorithms and compare them, like a complete immersive playground for research and content creation.

<p align="center">
<img src="ResourcesForREADME/TemporaryCoverImage.jpg" style="width:75%;"/>
</p>

For now though, the interface works fine, but first algorithm SAC, does not work too well. I'm building it in the most universal way possible, so I'm handling 3D observations (visual) and 1D observations (vector) with both continuous and multidiscrete actions in one algorithm. And the algorithm runs. No matter the action space or observation space, it will always run without errors and make random actions, but it doesn't learn well for now.

Continuous action with optimization works well, that's how I know at least that memory, buffers, interface and most of the code works as intended. Unfortunately, multidiscrete action agents don't learn well yet, even thought I thoroughly followed the guide. I also tried to implement multidiscrete SAC with only discrete reference, so I might be missing something.

For more details regarding the problem I will make a document with helpful materials.

## How to use

If you'd like to help, here's how it works for now:

Install mlagents_envs library to interface with Unity env builds.

```bash
pip install --no-dependencies mlagents_envs
```
No dependencies for now, as there's a version mismatch, but it should work just fine.

We support mostly Windows and Linux for now with server env builds, but it's easy to add new builds or use Unity Editor directly.

To run the script, simply uncomment your desired environment and comment the rest, so they don't interfere. None environment is for working directly with Unity Editor.

<p align="center">
<img src="ResourcesForREADME/EnvCode.jpg" style="width:50%;"/>
</p>

What are the environments?
- 3DBall is an environment with 1D observations and 2 continuous actions to balance a ball on a cube
- PushBlock is an environment with 1D observation and 1 discrete branch with 7 options to choose from.
- WallJump is an environment with 1D observations and 4 discrete branches with (3, 3, 3, 2) options.

I recommend using test.ipynb notebook with additional graphs and easier ability to inspect all the variables.
