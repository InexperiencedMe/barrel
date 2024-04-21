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
    observationBuffer[behavior] = env.getInitialObservations(behavior)
    

totalSteps = 10
decisionSteps, terminalSteps = env.getSteps(behavior)
for i in range(1, totalSteps+1):
    for behavior in behaviorNames:
        # print(f"Agents that request an action {len(decisionSteps)}. Agents that finished: {len(terminalSteps)}")
        # print(f"Agents that request an action {agentsThatRequestAction}. Agents that finished: {agentsThatFinished}")
        
        behaviorActionsForThisStep = {}
        specs = env.getSpecs(behavior)
        nrOfContinuousActions = specs["ContinuousActions"]
        nrOfDiscreteActions = len(specs["DiscreteActions"])
        if nrOfContinuousActions > 0:
            behaviorActionsForThisStep["continuous"] = np.zeros((len(decisionSteps), nrOfContinuousActions), dtype=np.float32)
        if nrOfDiscreteActions > 0:
            behaviorActionsForThisStep["discrete"] = np.zeros((len(decisionSteps), nrOfDiscreteActions), dtype=np.int32)


        if len(decisionSteps) > 0:
            observations = []
            for i, agent in enumerate(decisionSteps):
                print(f"Checking agent {agent} in from Agent that request an action {list(decisionSteps)} and we want to add its reward {decisionSteps[agent].reward} to our tensor rewards of size {rewards[behavior].shape}")
                rewards[behavior][agent] += decisionSteps[agent].reward 
                observations.append(decisionSteps[agent].obs)
                if nrOfContinuousActions > 0:
                    continuousAction, _, _, _ = agents[behavior].getContinuousActionAndValue(decisionSteps[agent].obs)
                    actionsBuffer[behavior]['continuous'][agent] = continuousAction
                    behaviorActionsForThisStep['continuous'][i] = continuousAction
                if nrOfDiscreteActions > 0:
                    discreteAction = agents[behavior].getDiscreteActionAndValue(decisionSteps[agent].obs)
                    actionsBuffer[behavior]['discrete'][agent] = discreteAction
                    behaviorActionsForThisStep['discrete'][i] = discreteAction
                    

                

            # obsShapes = []
            # obsDimensionalites = []
            # for element in observations[0]:
            #     obsShapes.append(element.shape)
            #     obsDimensionalites.append(len(element.shape))
            # print(f"Observation of shapes {obsShapes}, thus, dimensions {obsDimensionalites}\n")

            # print(f"Continuous action {agents[behavior].getContinuousActionAndValue(observations[0])[0]}")
            # print(f"Discrete action {agents[behavior].getDiscreteActionAndValue(observations[0])[0]}")
            for agent in terminalSteps:
                rewards[behavior][agent] += terminalSteps[agent].reward
                print(f"Final reward: {rewards[behavior][agent]:.2f}")
                rewards[behavior][agent] = 0
                memory[behavior].pushSingle(observationBuffer[behavior][agent], actionsBuffer[behavior][agent], terminalSteps[agent].reward, True, terminalSteps[agent].obs)
                observationBuffer[behavior][agent] = terminalSteps[agent].obs


            env.setActions(behavior, behaviorActionsForThisStep['continuous'], behaviorActionsForThisStep['discrete'])
    env.step()
    
    for behavior in behaviorNames:
        decisionSteps, terminalSteps = env.getSteps(behavior)
        for agent in decisionSteps:
            memory[behavior].pushSingle(observationBuffer[behavior][agent], actionsBuffer[behavior][agent], decisionSteps[agent].reward, False, decisionSteps[agent].obs)
            observationBuffer[behavior][agent] = decisionSteps[agent].obs
        for agent in terminalSteps:
            memory[behavior].pushSingle(observationBuffer[behavior][agent], actionsBuffer[behavior][agent], terminalSteps[agent].reward, True, terminalSteps[agent].obs)
            observationBuffer[behavior][agent] = terminalSteps[agent].obs
env.close()