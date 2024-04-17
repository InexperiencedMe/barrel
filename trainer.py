import numpy as np
import torch
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(linewidth=100, precision=2, sci_mode=False)
np.set_printoptions(linewidth=100, precision=2, suppress=True)

env = UnityInterface(None)
print(f"{env.getSpecs()}")
behaviorNames = env.getBehaviorNames()

totalSteps = 2
for i in range(1, totalSteps+1):
    for behavior in behaviorNames:
        decisionSteps, terminalSteps = env.getSteps(behavior)
        print(f"{env.getSpecs(behavior)}")
        for agentThatNeedsDecision in decisionSteps:
            decisionSteps
        # Initialize earlier tensors for observations for each agent
        # initialize tensors for rewards of each agent
        # handle memory


        
env.close()