import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import *
# torch.set_printoptions(linewidth=100, precision=4, sci_mode=False, threshold=2000)

# TODO: If I substituted every dictionary usage with named_tuple, would that be much faster and worth the effort?

seed: int = 2
torch_deterministic: bool = True
totalTimesteps: int = 100
graph = False
saveOnnx = True
onnxNameSuffix = "gen 1 corner overtrained"
# checkpointName = f"checkpoints\\3DBallHard-mainBranch-100000.pth"
# checkpointName = f"checkpoints\\Worm-newRun-2000000.pth"
checkpointName = f"checkpoints\\3DBall--50000.pth"
# checkpointName = f"checkpoints\\Crawler stable-50000.pth"

def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = torch_deterministic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# env = UnityInterface("Builds\\Windows\\Ball3D\\UnityEnvironment", seed=seed)      # 1D obs only, continuous action of size 2. Rewards: 0.1 for every step, -1 for fail, 100 is the max episodic return
# env = UnityInterface("Builds\\Windows\\Crawler\\UnityEnvironment", seed=seed)     # 1D obs only, continuous action of size 8
# env = UnityInterface("Builds\\Windows\\PushBlock\\UnityEnvironment", seed=seed)   # 1D obs only, discrete action of size (7). Rewards: 5 for win, -0.001 for every step
# env = UnityInterface("Builds\\Windows\\WallJump\\UnityEnvironment", seed=seed)    # 1D obs only, discrete action of size (3, 3, 3, 2)

# env = UnityInterface("Builds/Linux/Ball3D/Ball3D", seed=seed)                     # 1D obs only, continuous action of size 2. Rewards: 0.1 for every step, -1 for fail, 100 is the max episodic return
# env = UnityInterface("Builds/Linux/PushBlock/PushBlock", seed=seed)               # 1D obs only, discrete action of size (7). Rewards: 5 for win, -0.001 for every step
# env = UnityInterface("Builds/Linux/WallJump/WallJump", seed=seed)                 # 1D obs only, discrete action of size (3, 3, 3, 2)

env = UnityInterface(None, seed=seed) 

print(f"{env.getSpecs()}")
behaviorNames = env.getBehaviorNames()

totalAgentsCounts = 10000
for behavior in behaviorNames:
    totalAgentsCounts += (env.getSpecs(behavior)["AgentsCount"])

actor, actorOptimizer, memory, observationBuffer, actionsBuffer, = {}, {}, {}, {}, {}
QFunction1, QFunction2, QFunction1Target, QFunction2Target, criticOptimizer = {}, {}, {}, {}, {}
targetEntropyC, logAlphaC, alphaC, alphaOptimizerC = {}, {}, {}, {}
targetEntropyD, logAlphaD, alphaD, alphaOptimizerD = {}, {}, {}, {}
for behavior in behaviorNames:
    envSpecs = env.getSpecs(behavior)
    actor[behavior] = SAC(envSpecs).to(device)
    checkpoint = torch.load(checkpointName)
    actor[behavior].load_state_dict(checkpoint["actor"])

observationBuffer = [None] * totalAgentsCounts
for i in range(totalAgentsCounts):
    actionsBuffer[i] = {'continuous': None, 'discrete': None}
rewards = np.zeros(totalAgentsCounts)

finalRewards, criticLosses, actorLosses, alphaLosses, alphasC, alphasD, QEvaluations, logProbs = [], [], [], [], [], [], [], []
for globalStep in range(1, totalTimesteps+1):
    for behavior in behaviorNames:
        decisionSteps, terminalSteps = env.getSteps(behavior)
        observationsThatNeedAction = []
        for agent in decisionSteps:
            observation = decisionSteps[agent].obs
            observationsThatNeedAction.append(observation)
            reward = decisionSteps[agent].reward
            observationBuffer[agent] = observation
            rewards[agent] += reward
            # for obs in observation:
            #     print(f"obs element: {obs} of shape {obs.shape}")
            # if agent == 0:
            #     print(f"Agent 0 got reward: {reward}")
            
        for agent in terminalSteps:
            observation = terminalSteps[agent].obs
            reward = terminalSteps[agent].reward
            finalRewards.append(rewards[agent])
            # if agent == 0:
            #     print(f"Agent 0 got terminal reward: {reward}")
            rewards[agent] = 0

        
        behaviorActionsForThisStep = {}
        specs = env.getSpecs(behavior)
        nrOfContinuousActions = specs["ContinuousActions"] # TODO: Substitute it with actors[behavior].usingContinuousActions
        nrOfDiscreteActions = len(specs["DiscreteActions"])

        behaviorActionsForThisStep["continuous"] = torch.zeros((len(decisionSteps), nrOfContinuousActions), requires_grad=False, dtype=torch.float32, device=device)
        behaviorActionsForThisStep["discrete"] = torch.zeros((len(decisionSteps), nrOfDiscreteActions), requires_grad=False, dtype=torch.long, device=device)
        if len(observationsThatNeedAction) > 0:
            if actor[behavior].usingContinuousActions:
                behaviorActionsForThisStep["continuous"], _ = actor[behavior].getContinuousAction(observationsThatNeedAction, withLogProbs=False)
            if actor[behavior].usingDiscreteActions:
                behaviorActionsForThisStep["discrete"], _, _ = actor[behavior].getDiscreteAction((observationsThatNeedAction), withLogProbs=False)
        env.setActions(behavior, behaviorActionsForThisStep['continuous'].detach().cpu().numpy(), behaviorActionsForThisStep['discrete'].detach().cpu().numpy())
    env.step()  
env.close()

if graph:
    averagingNrR = 1
    beginning = 1
    dif = ((len(finalRewards) - beginning) % averagingNrR) + averagingNrR
    plt.style.use('seaborn-v0_8-bright')
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(25.6, 14.4)
    ax.plot(torch.tensor(finalRewards[beginning:-dif]).view(-1, averagingNrR).mean(-1))
    ax.set_title("Final Rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if saveOnnx:
    for behavior in behaviorNames:
        exportONNX(f"onnx\\{behavior[:behavior.find('?')]} {onnxNameSuffix}", actor[behavior], env.getSpecs(behavior))