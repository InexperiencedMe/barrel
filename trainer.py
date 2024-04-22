import numpy as np
import torch
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(linewidth=100, precision=2, sci_mode=False)
np.set_printoptions(linewidth=100, precision=2, suppress=True)

env = UnityInterface(None)
print(f"{env.getSpecs()}")
behaviorNames = env.getBehaviorNames()
agents, memory, rewards, observationBuffer, actionsBuffer = {}, {}, {}, {}, {}
for behavior in behaviorNames:
    agents[behavior] = PPO(env.getSpecs(behavior))
    memory[behavior] = Memory(5)
    rewards[behavior] = np.zeros((env.getSpecs(behavior)["AgentsCount"]))
    observationBuffer[behavior] = [None] * env.getSpecs(behavior)["AgentsCount"]
    actionsBuffer[behavior] = {}
    for i in range(env.getSpecs(behavior)["AgentsCount"]):
        actionsBuffer[behavior][i] = {'continuous': None, 'discrete': None}
    observationBuffer[behavior] = env.getInitialObservations(behavior, observationBuffer)
    

totalSteps = 10
for i in range(1, totalSteps+1):
    decisionSteps, terminalSteps = env.getSteps(behavior)
    for behavior in behaviorNames:

        observationsThatNeedAction = []
        for agent in decisionSteps:
            observation = decisionSteps[agent].obs
            observationsThatNeedAction.append(observation)
            reward = decisionSteps[agent].reward
            lastObservation = observationBuffer[behavior][agent]
            # BUG KeyError: 17 for SoccerTwos. HOW TO HANDLE DIFFERENT TEAMS? Unity gives them IDs together
            lastAction = actionsBuffer[behavior][agent]
            if lastObservation != None and lastAction["discrete"] != None and lastAction["discrete"] != None:
                memory[behavior].push(lastObservation, lastAction, reward, False, observation)
                observationBuffer[behavior][agent] = observation
            rewards[behavior][agent] += reward
            
        for agent in terminalSteps:
            observation = terminalSteps[agent].obs
            reward = terminalSteps[agent].reward
            lastObservation = observationBuffer[behavior][agent]
            lastAction = actionsBuffer[behavior][agent]
            if lastObservation != None and lastAction["discrete"] != None and lastAction["discrete"] != None:
                memory[behavior].push(lastObservation, lastAction, reward, True, observation)
                observationBuffer[behavior][agent] = None
            rewards[behavior][agent] += reward
            print(f"Final reward: {rewards[behavior][agent]}")
            rewards[behavior][agent] = 0

        
        behaviorActionsForThisStep = {}
        specs = env.getSpecs(behavior)
        nrOfContinuousActions = specs["ContinuousActions"]
        nrOfDiscreteActions = len(specs["DiscreteActions"])
        behaviorActionsForThisStep["continuous"] = np.zeros((len(decisionSteps), nrOfContinuousActions), dtype=np.float32)
        behaviorActionsForThisStep["discrete"] = np.zeros((len(decisionSteps), nrOfDiscreteActions), dtype=np.int32)

        # Doing it together to maybe do a one batched pass one day
        # Also need to stop the split for continuous and discrete. Model should spit out total action
        for i, agent in enumerate(decisionSteps):
            if nrOfContinuousActions > 0:
                continuousAction, _, _, _ = agents[behavior].getContinuousActionAndValue(observationsThatNeedAction[i])
                actionsBuffer[behavior][agent]['continuous'] = continuousAction.detach().numpy()
                behaviorActionsForThisStep['continuous'][i] = continuousAction.detach().numpy()
            if nrOfDiscreteActions > 0:
                discreteAction = agents[behavior].getDiscreteActionAndValue(observationsThatNeedAction[i])
                actionsBuffer[behavior][agent]['discrete'] = discreteAction.detach().numpy()
                behaviorActionsForThisStep['discrete'][i] = discreteAction.detach().numpy()
                    
            # obsShapes = []
            # obsDimensionalites = []
            # for element in observationsThatNeedAction[0]:
            #     obsShapes.append(element.shape)
            #     obsDimensionalites.append(len(element.shape))
            # print(f"Observation of shapes {obsShapes}, thus, dimensions {obsDimensionalites}\n")

            print(f"Completed a whole step. Continuous actions: {behaviorActionsForThisStep['continuous']}, Discrete actions: {behaviorActionsForThisStep['discrete']}")
            env.setActions(behavior, behaviorActionsForThisStep['continuous'], behaviorActionsForThisStep['discrete'])
    env.step()
env.close()