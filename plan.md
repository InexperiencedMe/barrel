
- imports
- getParams()

env = UnityInterface()
specs = env.getSpecs() # Returns dict with agents and their specs as another dict

We'll need setActions, step, reset, and all these things like converting actions to ActionTuple before passing. Since we dont want to think about this in main script

These will be just specs provided to the actor as a dict! DAMN
    strikerObservationSpace = specs["Striker"]["ObsSpace"]
    strikerContActionSize = specs["Striker"]["ContinuousActionSize"]
    strikerDiscreteActionSize = specs["Striker"]["DiscreteActionSize"]

- initialize nets. They should have builtin q networks as a part of the same architecture.
- Optimizers also should be inside. Optimizer will be a part of architecture, hidden in the API
- SAC will have automatic entropy tuning inside. As few magic numbers as possible
- Memory should also be instantiated inside, but.. Hmmm.. Yeah, need names, but they will be constant! (Observation, action, reward, dones, next observation)

(For architectures, a ton of things to think about. How to handle both vectors obs and visual obs. The whole point of visual obs is the spatial relationship, thats what convolutional nets are made for. I cannot just flatten it and "there you go")

class actor(nn.Module):
    def __init__(self, specs[agentName]) # Define layers, crtics, optimizers and memory
    def forward() # get Obs and produce continuous and discrete actions. We will convert it to ActionTuple before passing to Unity env
    def optimizePolicy()
    def optimizeQNets()
    def optimizeEntropy()
    def addMemory()
    def evaluateState() for q networks

