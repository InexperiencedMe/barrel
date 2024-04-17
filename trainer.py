import numpy as np
import torch
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(linewidth=100, precision=2, sci_mode=False)
np.set_printoptions(linewidth=100, precision=2, suppress=True)

env = UnityInterface(None)
print(f"{env.getSpecs()}")
behaviorNames = env.getBehaviorNames()

actions = {}
for behavior in behaviorNames:
    behaviorActions = {"continuous": None, "discrete": None}
    specs = env.getSpecs(behavior)
    nrOfContinuousActions = specs["ContinuousActions"]
    nrOfDiscreteActions = len(specs["DiscreteActions"])
    if nrOfContinuousActions > 0:
        behaviorActions["continuous"] = np.zeros((specs["AgentsCount"], nrOfContinuousActions), dtype=np.float32)
    if nrOfDiscreteActions > 0:
        behaviorActions["discrete"] = np.zeros((specs["AgentsCount"], nrOfDiscreteActions), dtype=np.int32)
    actions[behavior] = behaviorActions
    print(f"For behavior {behavior} we made actions dict = {behaviorActions}")    
print(f"Full initialized actions for all agents: {actions}")    

totalSteps = 200
for i in range(1, totalSteps+1):
    for behavior in behaviorNames:
        decisionSteps, terminalSteps = env.getSteps(behavior)
        print(f"Number of agents that request an action {len(decisionSteps)}")
        # if len(terminalSteps) > 0:
        #     env.reset()
        #     break
        env.setActions(behavior, actions[behavior]['continuous'], actions[behavior]['discrete'])
        env.step()
env.close()