import numpy as np
import torch
import time
from utils import *
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(linewidth=100, precision=4, sci_mode=False)
np.set_printoptions(linewidth=100, precision=4, suppress=True)

env = UnityInterface("Builds\\Ball3D\\UnityEnvironment")
# env = UnityInterface("Builds\\Crawler\\UnityEnvironment")
# env = UnityInterface(None)
print(f"{env.getSpecs()}")
behaviorNames = env.getBehaviorNames()
targetEntropy, logAlpha, alpha, alphaOptimizer = {}, {}, {}, {}

agents, QNet, memory, rewards, observationBuffer, actionsBuffer, = {}, {}, {}, {}, {}, {}
totalAgentsCounts = 0
for behavior in behaviorNames:
    totalAgentsCounts += (env.getSpecs(behavior)["AgentsCount"])
    
# NOTE: rewards, observations and actions are common for all behaviors
rewards = np.zeros(totalAgentsCounts)
observationBuffer = [None] * totalAgentsCounts
for i in range(totalAgentsCounts):
    actionsBuffer[i] = {'continuous': None, 'discrete': None}

for behavior in behaviorNames:
    agents[behavior] = SAC(env.getSpecs(behavior)).to(device)
    for name, params in agents[behavior].named_parameters():
        print(f"STARTING agent parameters {name}: {params}")
    QNet[behavior] = SoftQNetwork(env.getSpecs(behavior))
    memory[behavior] = Memory(10000)

    assert agents[behavior].usingContinuousActions or agents[behavior].usingDiscreteActions, "Agent not using continuous nor discrete actions, VERY BAD"


    # alpha[behavior] = 0.5
    targetEntropy[behavior] = torch.tensor((0), dtype=torch.float, device=device)
    divider = 0
    if agents[behavior].usingDiscreteActions:
        targetEntropy[behavior] -= torch.log(1 / sum(torch.tensor(env.getSpecs(behavior)["DiscreteActions"]))).to(device)
        divider += 1
    if agents[behavior].usingContinuousActions:
        targetEntropy[behavior] -= torch.tensor(env.getSpecs(behavior)["ContinuousActions"]).to(device)
        divider += 1
    targetEntropy[behavior] /= divider
    print(f"Target entropy: {targetEntropy}")

    logAlpha[behavior] = torch.zeros(1, requires_grad=True, device=device)
    alpha[behavior] = logAlpha[behavior].exp().item()
    alphaOptimizer[behavior] = optim.Adam([logAlpha[behavior]], lr=1e-3)

gamma = 0.99
batchSize = 32
actorUpdateInterval = 1
actorUpdateNumber = 1
qnetsUpdateFrequency = 1
entropyUpdateInterval = 1
finalRewards = []
qnetsLosses = []
actorLosses = []
alphaLosses = []
alphas = []
QEvaluations = []
logProbs = []
totalSteps = 30000
for i in range(1, totalSteps+1):
    # startInference = time.time()
    for behavior in behaviorNames:
        decisionSteps, terminalSteps = env.getSteps(behavior)
        # print(f"For behavior {behavior} in step {i}/{totalSteps} we have decisionSteps agents {list(decisionSteps)} and terminal steps {list(terminalSteps)}")
        observationsThatNeedAction = []
        for agent in decisionSteps:

            observation = decisionSteps[agent].obs
            observationsThatNeedAction.append(observation)
            reward = decisionSteps[agent].reward
            # print(f"agent {agent}, observationBuffer: {observationBuffer} of len {len(observation)}")
            lastObservation = observationBuffer[agent]
            lastActionContinuous = actionsBuffer[agent]['continuous']
            lastActionDiscrete = actionsBuffer[agent]['discrete']
            # print(f"\n\nDECISION step {i}: For agent {agent}:\nLastObs: {lastObservation},\nCurrentObs: {observation},\nLastActionC: {lastActionContinuous},\nCurrentReward:{reward}\n")
            if lastObservation != None and (lastActionContinuous != None or lastActionDiscrete != None):
                memory[behavior].push(lastObservation, lastActionContinuous, lastActionDiscrete, reward, False, observation)
            observationBuffer[agent] = observation
            rewards[agent] += reward
            
        for agent in terminalSteps:
            observation = terminalSteps[agent].obs
            reward = terminalSteps[agent].reward
            lastObservation = observationBuffer[agent]
            lastActionContinuous = actionsBuffer[agent]['continuous']
            lastActionDiscrete = actionsBuffer[agent]['discrete']
            # print(f"TERMINAL step {i}: For agent {agent}:\nLastObs: {lastObservation},\nCurrentObs: {observation},\nCurrentReward:{reward}\n")
            # Technically could skip the action None check. If lastObs exist, action does too
            if lastObservation != None and (lastActionContinuous != None or lastActionDiscrete != None):
                memory[behavior].push(lastObservation, lastActionContinuous, lastActionDiscrete, reward, True, observation)
                observationBuffer[agent] = None
                actionsBuffer[agent]['continuous'] = None
                actionsBuffer[agent]['discrete'] = None
                # Save rewards only if we made an action before, otherwise the initial state was terminated state
                rewards[agent] += reward
                # if rewards[agent] > 4:
                #     print(f"Final reward: {rewards[agent]:>.2f}")
            finalRewards.append(rewards[agent])
            rewards[agent] = 0

        
        behaviorActionsForThisStep = {}
        specs = env.getSpecs(behavior)
        nrOfContinuousActions = specs["ContinuousActions"] # TODO: Substitute it with actors[behavior].usingContinuousActions
        nrOfDiscreteActions = len(specs["DiscreteActions"])
        behaviorActionsForThisStep["continuous"] = torch.zeros((len(decisionSteps), nrOfContinuousActions), requires_grad=False, dtype=torch.float32, device=device)
        behaviorActionsForThisStep["discrete"] = torch.zeros((len(decisionSteps), nrOfDiscreteActions), requires_grad=False ,dtype=torch.int32, device=device)

        # Batched pass to get actions  
        if len(observationsThatNeedAction) > 0:
            if agents[behavior].usingContinuousActions:
                behaviorActionsForThisStep['continuous'], _ = agents[behavior].getContinuousActionAndValue(observationsThatNeedAction, withLogprobs=False)
            if agents[behavior].usingDiscreteActions:
                behaviorActionsForThisStep['discrete'], _ = agents[behavior].getDiscreteActionAndValue(observationsThatNeedAction)

        # Transcribe the actions to buffer
            for j, agent in enumerate(decisionSteps):
                if nrOfContinuousActions > 0:
                    actionsBuffer[agent]['continuous'] =  behaviorActionsForThisStep['continuous'][j]
                if nrOfDiscreteActions > 0:
                    actionsBuffer[agent]['discrete'] = behaviorActionsForThisStep['discrete'][j]
                
            # obsShapes = []
            # obsDimensionalites = []
            # for element in observationsThatNeedAction[0]:
            #     obsShapes.append(element.shape)
            #     obsDimensionalites.append(len(element.shape))
            # print(f"Observation of shapes {obsShapes}, thus, dimensions {obsDimensionalites}\n")

        # print(f"Setting Continuous actions: {behaviorActionsForThisStep['continuous']}, Discrete actions: {behaviorActionsForThisStep['discrete']}")
        env.setActions(behavior, behaviorActionsForThisStep['continuous'].detach().cpu().numpy(), behaviorActionsForThisStep['discrete'].detach().cpu().numpy())
    env.step()
    # endInference = time.time()
    # inferenceTime = endInference-startInference

    # startOptimization = time.time()
    for behavior in behaviorNames:
        # print(f"memory length for behavior {behavior}: {len(memory[behavior])}")
        if len(memory[behavior]) > batchSize:
            mem = memory[behavior].sample(batchSize)
            # Critic update
            # print(f"Sampled experiences:{mem}")
            # startQnetsOptim = time.time()
            with torch.no_grad():
                nextStateActionsContinuous, nextStateActionsDiscrete, nextStateLogProbsContinuous, nextStateLogProbsDiscrete, divider = None, None, 0, 0, 0
                
                if agents[behavior].usingContinuousActions:
                    nextStateActionsContinuous, nextStateLogProbsContinuous = agents[behavior].getContinuousActionAndValue(mem.nextObservations)
                    divider += 1
                if agents[behavior].usingDiscreteActions:
                    nextStateActionsDiscrete, nextStateLogProbsDiscrete = agents[behavior].getDiscreteActionAndValue(mem.nextObservations)
                    divider += 1

                QFunction1NextTarget = QNet[behavior].QNet1Target(mem.nextObservations, nextStateActionsContinuous, nextStateActionsDiscrete)
                QFunction2NextTarget = QNet[behavior].QNet2Target(mem.nextObservations, nextStateActionsContinuous, nextStateActionsDiscrete)
                minQNextTarget = torch.min(QFunction1NextTarget, QFunction2NextTarget).view(-1) - alpha[behavior] * (nextStateLogProbsContinuous + nextStateLogProbsDiscrete) / divider
                nextQValue = torch.tensor(mem.rewards, device=device, dtype=torch.float32) + torch.logical_not(torch.tensor(mem.dones, device=device)) * gamma * minQNextTarget
                
            QFunction1ActionValues = QNet[behavior].QNet1(mem.observations, torch.stack(mem.actionsContinuous).to(device).detach() if agents[behavior].usingContinuousActions else None, torch.stack(mem.actionsDiscrete).detach() if agents[behavior].usingDiscreteActions else None).view(-1)
            QFunction2ActionValues = QNet[behavior].QNet2(mem.observations, torch.stack(mem.actionsContinuous).to(device).detach() if agents[behavior].usingContinuousActions else None, torch.stack(mem.actionsDiscrete).detach() if agents[behavior].usingDiscreteActions else None).view(-1)
            QFunction1Loss = F.mse_loss(QFunction1ActionValues, nextQValue)
            QFunction2Loss = F.mse_loss(QFunction2ActionValues, nextQValue)
            QFunctionsTotalLoss = QFunction1Loss + QFunction2Loss
            
            QNet[behavior].QNetsOptimizer.zero_grad()
            QFunctionsTotalLoss.backward()
            QNet[behavior].QNetsOptimizer.step()
            # endQnetsOptim = time.time()
            # qnetsOptimTime = endQnetsOptim-startQnetsOptim

            # # Actor update
            if i % actorUpdateInterval == 0:
                for j in range(actorUpdateNumber):
                    # startActorOptim = time.time()
                    if j > 0:
                        mem = memory[behavior].sample(batchSize)
                    stateActionsContinuous, stateActionsDiscrete, stateLogProbsContinuous, stateLogProbsDiscrete, divider = None, None, torch.tensor(0, device=device), torch.tensor(0, device=device), torch.tensor(0, device=device)
                    
                    if agents[behavior].usingContinuousActions:
                        stateActionsContinuous, stateLogProbsContinuous = agents[behavior].getContinuousActionAndValue(mem.observations)
                        stateActionsContinuous = stateActionsContinuous.detach()
                        divider += 1
                    if agents[behavior].usingDiscreteActions:
                        stateActionsDiscrete, stateLogProbsDiscrete = agents[behavior].getDiscreteActionAndValue(mem.observations)
                        stateActionsDiscrete = stateActionsDiscrete.detach()
                        divider += 1

                    QFunction1Evaluation = QNet[behavior].QNet1(mem.observations, stateActionsContinuous, stateActionsDiscrete)
                    QFunction2Evaluation = QNet[behavior].QNet2(mem.observations, stateActionsContinuous, stateActionsDiscrete)

                    minQEvaluation = torch.min(QFunction1Evaluation, QFunction2Evaluation)
                    # print(f"minQeval of shape: {minQEvaluation.shape}: {minQEvaluation}")
                    # print(f"Actor loss: entropy - currentevaluation: {(alpha[behavior] * (stateLogProbsContinuous + stateLogProbsDiscrete) / divider)} - {minQEvaluation} and we mean it into {((alpha[behavior] * (stateLogProbsContinuous + stateLogProbsDiscrete) / divider) - minQEvaluation).mean()}")
                    actorLoss = ((alpha[behavior] * (stateLogProbsContinuous + stateLogProbsDiscrete) / divider) - minQEvaluation).mean()

                    

                    agents[behavior].actorOptimizer.zero_grad()
                    actorLoss.backward()
                    agents[behavior].actorOptimizer.step()
                    
            # update the target network
            if i % qnetsUpdateFrequency == 0:
                for _ in range(qnetsUpdateFrequency):
                    for param, targetParam in zip(QNet[behavior].QNet1.parameters(), QNet[behavior].QNet1Target.parameters()):
                        targetParam.data.copy_(QNet[behavior].tau*param.data + (1 - QNet[behavior].tau)*targetParam.data)
                    for param, targetParam in zip(QNet[behavior].QNet2.parameters(), QNet[behavior].QNet2Target.parameters()):
                        targetParam.data.copy_(QNet[behavior].tau*param.data + (1 - QNet[behavior].tau)*targetParam.data)
            
            if i % entropyUpdateInterval == 0:
                for j in range(actorUpdateNumber):
                    # startActorOptim = time.time()
                    if j > 0:
                        mem = memory[behavior].sample(batchSize)
                    with torch.no_grad():
                        logProbabilitiesC, logProbabilitiesD, divider = torch.tensor(0, device=device), torch.tensor(0, device=device), 0
                        if agents[behavior].usingContinuousActions:
                            _, logProbabilitiesC = agents[behavior].getContinuousActionAndValue(mem.observations)
                            divider += 1
                        if agents[behavior].usingDiscreteActions:
                            _, logProbabilitiesD = agents[behavior].getDiscreteActionAndValue(mem.observations)
                            divider += 1
                    alphaLoss = (-logAlpha[behavior].exp()*((logProbabilitiesC.to(device) + logProbabilitiesD.to(device)) / divider + targetEntropy[behavior])).mean()

                    alphaOptimizer[behavior].zero_grad()
                    alphaLoss.backward()
                    alphaOptimizer[behavior].step()
                    alpha[behavior] = logAlpha[behavior].exp().item()

            if i % 1000 == 0:
                # print(f"Step {i}, i % 100 = {i%100}, so we're here")
                # print(f"Step {i}, Alpha: {alpha[behavior]:>4.2f}, Alpha Loss: {alphaLoss:>6.2f} Actor loss: {actorLoss:>8.2f}, QF loss: {QFunctionsTotalLoss:>8.2f}")
                print(f"Step {i}, Actor loss: {actorLoss:>8.2f}, QF loss: {QFunctionsTotalLoss:>8.2f}")

            if i % 1 == 0:
                qnetsLosses.append(QFunctionsTotalLoss)
                actorLosses.append(actorLoss)
                # alphaLosses.append(alphaLoss)
                alphas.append(alpha[behavior])
                QEvaluations.append(minQEvaluation.view(-1).mean())
                logProbs.append(((stateLogProbsContinuous + stateLogProbsDiscrete) / divider).mean(-1))

        # endOptimization = time.time()
        # optimizationTime = endOptimization-startOptimization
        # print(f"{1/(optimizationTime+0.00001):>4.1f} optimizations per second")
        # print(f"{1/optimizationTime:>4.1f} optimizations per second, {1/inferenceTime:>4.1f} inferences per second")

for behavior in behaviorNames:
    for name, params in agents[behavior].named_parameters():
        print(f"ENDING agent parameters {name}: {params}")
#     for name, params in QNet[behavior].QNet1.named_parameters():
#         print(f"ENDING QNET1 parameters {name}: {params}")
    

env.close()