import numpy as np
import torch
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(linewidth=100, precision=2, sci_mode=False)
np.set_printoptions(linewidth=100, precision=2, suppress=True)

env = UnityInterface(None)
print(f"{env.getSpecs()}")
behaviorNames = env.getBehaviorNames()


totalSteps = 1000
for i in range(1, totalSteps+1):
    for behavior in behaviorNames:
        decisionSteps, terminalSteps = env.getSteps(behavior)
        agentsThatRequestAction = list(decisionSteps)
        agentsThatFinished = list(terminalSteps)
        print(f"Number of agents that request an action {len(decisionSteps)}. Number of agents that finished: {len(terminalSteps)}")
        
        behaviorActions = {"continuous": None, "discrete": None}
        specs = env.getSpecs(behavior)
        nrOfContinuousActions = specs["ContinuousActions"]
        nrOfDiscreteActions = len(specs["DiscreteActions"])
        if nrOfContinuousActions > 0:
            behaviorActions["continuous"] = np.zeros((len(agentsThatRequestAction), nrOfContinuousActions), dtype=np.float32)
        if nrOfDiscreteActions > 0:
            behaviorActions["discrete"] = np.zeros((len(agentsThatRequestAction), nrOfDiscreteActions), dtype=np.int32)

        

        env.setActions(behavior, behaviorActions['continuous'], behaviorActions['discrete'])
        env.step()
env.close()