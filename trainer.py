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

totalAgentsCounts = 0
for behavior in behaviorNames:
    totalAgentsCounts += (env.getSpecs(behavior)["AgentsCount"])
    
# NOTE: rewards, observations and actions are common for all behaviors
rewards = np.zeros(totalAgentsCounts)
observationBuffer = [None] * totalAgentsCounts
for i in range(totalAgentsCounts):
    actionsBuffer[i] = {'continuous': None, 'discrete': None}
observationBuffer = env.getInitialObservations(observationBuffer)

for behavior in behaviorNames:
    agents[behavior] = PPO(env.getSpecs(behavior))
    memory[behavior] = Memory(5)
    

totalSteps = 100
for i in range(1, totalSteps+1):
    with torch.no_grad():
        for behavior in behaviorNames:
            decisionSteps, terminalSteps = env.getSteps(behavior)
            # print(f"For behavior {behavior} in step {i}/{totalSteps} we have decisionSteps agents {list(decisionSteps)} and terminal steps {list(terminalSteps)}")
            observationsThatNeedAction = []
            for agent in decisionSteps:
                observation = decisionSteps[agent].obs
                observationsThatNeedAction.append(observation)
                reward = decisionSteps[agent].reward
                lastObservation = observationBuffer[agent]
                lastAction = actionsBuffer[agent]
                if lastObservation != None and lastAction["discrete"] != None and lastAction["discrete"] != None:
                    memory[behavior].push(lastObservation, lastAction, reward, False, observation)
                    observationBuffer[agent] = observation
                rewards[agent] += reward
                
            for agent in terminalSteps:
                observation = terminalSteps[agent].obs
                reward = terminalSteps[agent].reward
                lastObservation = observationBuffer[agent]
                lastAction = actionsBuffer[agent]
                if lastObservation != None and lastAction["discrete"] != None and lastAction["discrete"] != None:
                    memory[behavior].push(lastObservation, lastAction, reward, True, observation)
                    observationBuffer[agent] = None
                    # Save rewards only if we made an action before, otherwise the initial state was terminated state
                    rewards[agent] += reward
                    print(f"Final reward: {rewards[agent]}")
                rewards[agent] = 0

            
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
                    actionsBuffer[agent]['continuous'] = continuousAction
                    behaviorActionsForThisStep['continuous'][i] = continuousAction
                if nrOfDiscreteActions > 0:
                    discreteAction, _, _, _ = agents[behavior].getDiscreteActionAndValue(observationsThatNeedAction[i])
                    actionsBuffer[agent]['discrete'] = discreteAction
                    behaviorActionsForThisStep['discrete'][i] = discreteAction
                        
                # obsShapes = []
                # obsDimensionalites = []
                # for element in observationsThatNeedAction[0]:
                #     obsShapes.append(element.shape)
                #     obsDimensionalites.append(len(element.shape))
                # print(f"Observation of shapes {obsShapes}, thus, dimensions {obsDimensionalites}\n")

            # print(f"Completed a whole step. Continuous actions: {behaviorActionsForThisStep['continuous']}, Discrete actions: {behaviorActionsForThisStep['discrete']}")
            env.setActions(behavior, behaviorActionsForThisStep['continuous'], behaviorActionsForThisStep['discrete'])
        env.step()



env.close()