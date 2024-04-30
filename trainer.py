import numpy as np
import torch
import time
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(linewidth=100, precision=2, sci_mode=False)
np.set_printoptions(linewidth=100, precision=2, suppress=True)

env = UnityInterface(None)
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
    QNet[behavior] = SoftQNetwork(env.getSpecs(behavior))
    memory[behavior] = Memory(10000)

    assert agents[behavior].usingContinuousActions or agents[behavior].usingDiscreteActions, "Agent not using continuous nor discrete actions, VERY BAD"

    targetEntropy[behavior] = torch.tensor((0), dtype=torch.float, device=device)
    if agents[behavior].usingDiscreteActions:
        targetEntropy[behavior] -= torch.log(1 / sum(torch.tensor(env.getSpecs(behavior)["DiscreteActions"]))).to(device)
    if agents[behavior].usingContinuousActions:
        targetEntropy[behavior] -= torch.tensor(env.getSpecs(behavior)["ContinuousActions"]).to(device)
    
    logAlpha[behavior] = torch.zeros(1, requires_grad=True, device=device)
    alpha[behavior] = logAlpha[behavior].exp().item()
    alphaOptimizer[behavior] = optim.Adam([logAlpha[behavior]], lr=1e-3)

# alpha = 0.2
gamma = 0.99
batchSize = 64
actorUpdateFrequency = 5
qnetsUpdateFrequency = 1

totalSteps = 100000
for i in range(1, totalSteps+1):
    # startInference = time.time()
    with torch.no_grad():
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
                if lastObservation != None and (lastActionContinuous != None or lastActionDiscrete != None):
                    memory[behavior].push(lastObservation, lastActionContinuous, lastActionDiscrete, reward, True, observation)
                    observationBuffer[agent] = None
                    # Save rewards only if we made an action before, otherwise the initial state was terminated state
                    rewards[agent] += reward
                    # print(f"Final reward: {rewards[agent]:>.2f}")
                rewards[agent] = 0

            
            behaviorActionsForThisStep = {}
            specs = env.getSpecs(behavior)
            nrOfContinuousActions = specs["ContinuousActions"] # TODO: Substitute it with actors[behavior].usingContinuousActions
            nrOfDiscreteActions = len(specs["DiscreteActions"])
            behaviorActionsForThisStep["continuous"] = torch.zeros((len(decisionSteps), nrOfContinuousActions), dtype=torch.float32, device=device)
            behaviorActionsForThisStep["discrete"] = torch.zeros((len(decisionSteps), nrOfDiscreteActions), dtype=torch.int32, device=device)

            # Batched pass to get actions  
            if len(observationsThatNeedAction) > 0:
                if agents[behavior].usingContinuousActions:
                    behaviorActionsForThisStep['continuous'], _ = agents[behavior].getContinuousActionAndValue(observationsThatNeedAction)
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
            env.setActions(behavior, behaviorActionsForThisStep['continuous'].cpu().numpy(), behaviorActionsForThisStep['discrete'].cpu().numpy())
        env.step()
        # endInference = time.time()
        # inferenceTime = endInference-startInference


    # print(f"memory length for behavior {behavior}: {len(memory[behavior])}")
    if len(memory[behavior]) > batchSize:
        # startOptimization = time.time()
        mem = memory[behavior].sample(min(len(memory[behavior]), batchSize))
        # print(f"Sampled experiences: {mem}")
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
            nextQValue = torch.tensor(mem.rewards, device=device) + torch.logical_not(torch.tensor(mem.dones, device=device)) * gamma * minQNextTarget.view(-1)
            
        QFunction1ActionValues = QNet[behavior].QNet1(mem.observations, torch.stack(mem.actionsContinuous).to(device) if agents[behavior].usingContinuousActions else None, torch.stack(mem.actionsDiscrete) if agents[behavior].usingDiscreteActions else None).view(-1)
        QFunction2ActionValues = QNet[behavior].QNet2(mem.observations, torch.stack(mem.actionsContinuous).to(device) if agents[behavior].usingContinuousActions else None, torch.stack(mem.actionsDiscrete) if agents[behavior].usingDiscreteActions else None).view(-1)
        QFunction1Loss = F.mse_loss(QFunction1ActionValues, nextQValue)
        QFunction2Loss = F.mse_loss(QFunction2ActionValues, nextQValue)
        QFunctionsTotalLoss = QFunction1Loss + QFunction2Loss
        
        QNet[behavior].QNetsOptimizer.zero_grad()
        QFunctionsTotalLoss.backward()
        QNet[behavior].QNetsOptimizer.step()
        # endQnetsOptim = time.time()
        # qnetsOptimTime = endQnetsOptim-startQnetsOptim

        # Actor update
        if i % actorUpdateFrequency == 0:
            for j in range(actorUpdateFrequency):
                # startActorOptim = time.time()
                if j > 0:
                    mem = memory[behavior].sample(min(len(memory[behavior]), batchSize))
                stateActionsContinuous, stateActionsDiscrete, stateLogProbsContinuous, stateLogProbsDiscrete, divider = None, None, torch.tensor(0, device=device), torch.tensor(0, device=device), torch.tensor(0, device=device)
                
                if agents[behavior].usingContinuousActions:
                    stateActionsContinuous, stateLogProbsContinuous = agents[behavior].getContinuousActionAndValue(mem.observations)
                    divider += 1
                if agents[behavior].usingDiscreteActions:
                    stateActionsDiscrete, stateLogProbsDiscrete = agents[behavior].getDiscreteActionAndValue(mem.observations)
                    divider += 1

                QFunction1Evaluation = QNet[behavior].QNet1(mem.observations, stateActionsContinuous, stateActionsDiscrete)
                QFunction2Evaluation = QNet[behavior].QNet2(mem.observations, stateActionsContinuous, stateActionsDiscrete)

                minQEvaluation = torch.min(QFunction1Evaluation, QFunction2Evaluation)
                actorLoss = ((alpha[behavior] * (stateLogProbsContinuous + stateLogProbsDiscrete) / divider) - minQEvaluation.view(-1)).mean()

                agents[behavior].actorOptimizer.zero_grad()
                actorLoss.backward()
                agents[behavior].actorOptimizer.step()
                # endActorOptim = time.time()
                # actorOptimTime = endActorOptim-startActorOptim


                # startEntropyOptim = time.time()
                with torch.no_grad():
                    logProbabilitiesC, logProbabilitiesD = torch.tensor(0, device=device), torch.tensor(0, device=device)
                    if agents[behavior].usingContinuousActions:
                        _, logProbabilitiesC = agents[behavior].getContinuousActionAndValue(mem.nextObservations)
                    if agents[behavior].usingDiscreteActions:
                        _, logProbabilitiesD = agents[behavior].getDiscreteActionAndValue(mem.nextObservations)
                alphaLoss = (-logAlpha[behavior].exp()*((logProbabilitiesC.to(device) + logProbabilitiesD.to(device)) / divider + targetEntropy[behavior])).mean()

                alphaOptimizer[behavior].zero_grad()
                alphaLoss.backward()
                alphaOptimizer[behavior].step()
                alpha[behavior] = logAlpha[behavior].exp().item()
                # endEntropyOptim = time.time()
                # entropyOptimTime = endEntropyOptim-startEntropyOptim
        # update the target network
        if i % qnetsUpdateFrequency == 0:
            for _ in range(qnetsUpdateFrequency):
                for param, targetParam in zip(QNet[behavior].QNet1.parameters(), QNet[behavior].QNet1Target.parameters()):
                    targetParam.data.copy_(QNet[behavior].tau*param.data + (1 - QNet[behavior].tau)*targetParam.data)
                for param, targetParam in zip(QNet[behavior].QNet2.parameters(), QNet[behavior].QNet2Target.parameters()):
                    targetParam.data.copy_(QNet[behavior].tau*param.data + (1 - QNet[behavior].tau)*targetParam.data)
        
        if i % 100 == 0:
            print(f"Step {i}, i % 100 = {i%100}, so we're here")
            print(f"Alpha: {alpha[behavior]:>8.2f}, Alpha loss: {alphaLoss:>8.2f}, Actor loss: {actorLoss:>8.2f}, QF loss: {QFunctionsTotalLoss:>8.2f}")
    # endOptimization = time.time()
    # optimizationTime = endOptimization-startOptimization
    # print(f"{1/optimizationTime:>4.1f} optimizations per second (QNETS: {1/qnetsOptimTime:>4.1f}, Actor: {1/actorOptimTime:>4.1f}, Entropy: {1/entropyOptimTime:>4.1f}), {1/inferenceTime:>4.1f} inferences per second")
    # print(f"{1/optimizationTime:>4.1f} optimizations per second, {1/inferenceTime:>4.1f} inferences per second")
env.close()