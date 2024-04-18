import numpy as np
import torch
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(linewidth=100, precision=2, sci_mode=False)
np.set_printoptions(linewidth=100, precision=2, suppress=True)

env = UnityInterface(None)
print(f"{env.getSpecs()}")
behaviorNames = env.getBehaviorNames()

rewards = {}
for behavior in behaviorNames:
    rewards[behavior] = np.zeros((env.getSpecs(behavior)["AgentsCount"]))

totalSteps = 10
for i in range(1, totalSteps+1):
    for behavior in behaviorNames:
        decisionSteps, terminalSteps = env.getSteps(behavior)
        agentsThatRequestAction = list(decisionSteps)
        agentsThatFinished = list(terminalSteps)
        # print(f"Agents that request an action {len(decisionSteps)}. Agents that finished: {len(terminalSteps)}")
        
        behaviorActions = {"continuous": None, "discrete": None} # This is agent's part
        specs = env.getSpecs(behavior)
        nrOfContinuousActions = specs["ContinuousActions"]
        nrOfDiscreteActions = len(specs["DiscreteActions"])
        if nrOfContinuousActions > 0:
            behaviorActions["continuous"] = np.zeros((len(agentsThatRequestAction), nrOfContinuousActions), dtype=np.float32)
        if nrOfDiscreteActions > 0:
            behaviorActions["discrete"] = np.zeros((len(agentsThatRequestAction), nrOfDiscreteActions), dtype=np.int32)


        observations = []
        for agent in agentsThatRequestAction: # rewards handling
            rewards[behavior][agent] += decisionSteps[agent].reward 
            observations.append(decisionSteps[agent].obs)
        
        obsShapes = []
        obsDimensionalites = []
        for element in observations[0]:
            obsShapes.append(element.shape)
            obsDimensionalites.append(len(element.shape))
        print(f"{observations[0]} of shapes {obsShapes}, thus, dimensions {obsDimensionalites}\n\n")

        for agent in agentsThatFinished:
            rewards[behavior][agent] += terminalSteps[agent].reward
            print(f"Final reward: {rewards[behavior][agent]:.2f}")
            rewards[behavior][agent] = 0

        env.setActions(behavior, behaviorActions['continuous'], behaviorActions['discrete'])
        env.step()
env.close()