# Barrel (BaRL)
## Raising the (lowest) bar of RL implementations

Development phase. Discrete action space doesn't work, continuous action space works just fine, but we haven't tested for more than 20k steps


If you'd like to help, here's how it works for now:

Install mlagents_envs library to interface with Unity env builds.

```bash
pip install --no-dependencies mlagents_envs
```

No dependencies for now, as there's a version mismatch, but it should work just fine.

We support mostly Windows and Linux for now with server env builds, but it's easy to add new builds or use Unity Editor directly.