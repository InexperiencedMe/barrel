import numpy as np
import torch
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(linewidth=100, precision=2, sci_mode=False)
np.set_printoptions(linewidth=100, precision=2, suppress=True)

env = UnityInterface(None)
print(f"{env.getSpecs()}")
behaviorNames = env.getBehaviorNames()
agents, QNet, memory, rewards, observationBuffer, actionsBuffer = {}, {}, {}, {}, {}, {}

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
    QNet[behavior] = SoftQNetwork(env.getSpecs(behavior))
    memory[behavior] = Memory(10)

alphaContinuous, alphaDiscrete = 0.2, 0.2
gamma = 0.99
# targetEntropy = -targetEntropyScale * torch.log(1 / torch.tensor(envs.single_action_space.n))
# logAlpha = torch.zeros(1, requires_grad=True, device=device)
# alpha = logAlpha.exp().item()
# alphaOptimizer = optim.Adam([logAlpha], lr=q_lr)


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
                lastActionContinuous = actionsBuffer[agent]['continuous']
                lastActionDiscrete = actionsBuffer[agent]['discrete']
                if lastObservation != None and (lastActionContinuous != None or lastActionDiscrete != None):
                    memory[behavior].push(lastObservation, lastActionContinuous, lastActionDiscrete, reward, True, observation)
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
                    print(f"Final reward: {rewards[agent]}")
                rewards[agent] = 0

            
            behaviorActionsForThisStep = {}
            specs = env.getSpecs(behavior)
            nrOfContinuousActions = specs["ContinuousActions"] # TODO: Substitute it with actors[behavior].usingContinuousActions
            nrOfDiscreteActions = len(specs["DiscreteActions"])
            behaviorActionsForThisStep["continuous"] = torch.zeros((len(decisionSteps), nrOfContinuousActions), dtype=torch.float32)
            behaviorActionsForThisStep["discrete"] = torch.zeros((len(decisionSteps), nrOfDiscreteActions), dtype=torch.int32)

            # Batched pass to get actions  
            if len(observationsThatNeedAction) > 0:
                if agents[behavior].usingContinuousActions:
                    behaviorActionsForThisStep['continuous'], _, _, = agents[behavior].getContinuousActionAndValue(observationsThatNeedAction)
                if agents[behavior].usingDiscreteActions:
                    behaviorActionsForThisStep['discrete'], _, _, _, = agents[behavior].getDiscreteActionAndValue(observationsThatNeedAction)

            # Transcribe the actions to buffer
                for i, agent in enumerate(decisionSteps):
                    if nrOfContinuousActions > 0:
                        actionsBuffer[agent]['continuous'] =  behaviorActionsForThisStep['continuous'][i]
                    if nrOfDiscreteActions > 0:
                        actionsBuffer[agent]['discrete'] = behaviorActionsForThisStep['discrete'][i]
                    
                # obsShapes = []
                # obsDimensionalites = []
                # for element in observationsThatNeedAction[0]:
                #     obsShapes.append(element.shape)
                #     obsDimensionalites.append(len(element.shape))
                # print(f"Observation of shapes {obsShapes}, thus, dimensions {obsDimensionalites}\n")

            # print(f"Completed a whole step. Continuous actions: {behaviorActionsForThisStep['continuous']}, Discrete actions: {behaviorActionsForThisStep['discrete']}")
            env.setActions(behavior, behaviorActionsForThisStep['continuous'].numpy(), behaviorActionsForThisStep['discrete'].numpy())
        env.step()



        print(f"memory length for behavior {behavior}: {len(memory[behavior])}")
        if len(memory[behavior]) > 2:
            batchSize = min(len(memory[behavior]), 64)
            print(f"Batch size: {batchSize}")
            sampledExperiences = memory[behavior].sample(min(len(memory[behavior]), 64))
            # print(f"Sampled experiences: {sampledExperiences}")
            # Critic update
            with torch.no_grad():
                if agents[behavior].usingContinuousActions and agents[behavior].usingDiscreteActions:
                    nextStateActionsContinuous, nextStateLogprobsContinuous, _ = agents[behavior].getContinuousActionAndValue(sampledExperiences.nextObservations)
                    nextStateActionsDiscrete, nextStateLogprobsDiscrete, nextStateProbsDiscrete, _ = agents[behavior].getDiscreteActionAndValue(sampledExperiences.nextObservations)
                    QFunction1NextTarget = QNet[behavior].QNet1Target(sampledExperiences.nextObservations, nextStateActionsContinuous)
                    QFunction2NextTarget = QNet[behavior].QNet2Target(sampledExperiences.nextObservations, nextStateActionsContinuous)
                    minQNextTarget = nextStateProbsDiscrete * (torch.min(QFunction1NextTarget, QFunction2NextTarget) - alphaContinuous * nextStateLogprobsContinuous - alphaDiscrete * nextStateLogprobsDiscrete)
                    nextQValue = torch.tensor(sampledExperiences.rewards) + torch.logical_not(torch.tensor(sampledExperiences.dones)) * gamma * (minQNextTarget.sum(1)).view(-1)
                
                    QFunction1ActionValues = QNet[behavior].QNet1(sampledExperiences.observations, sampledExperiences.actionsContinuous).gather(1, sampledExperiences.actionsDiscrete.long().view(-1, 1)).squeeze().view(-1)
                    QFunction2ActionValues = QNet[behavior].QNet2(sampledExperiences.observations, sampledExperiences.actionsContinuous).gather(1, sampledExperiences.actionsDiscrete.long().view(-1, 1)).squeeze().view(-1)

                elif agents[behavior].usingContinuousActions:
                    nextStateActionsContinuous, nextStateLogProbs, _ = agents[behavior].getContinuousActionAndValue(sampledExperiences.nextObservations)
                    QFunction1NextTarget = QNet[behavior].QNet1Target(sampledExperiences.next_observations, nextStateActionsContinuous)
                    QFunction2NextTarget = QNet[behavior].QNet2Target(sampledExperiences.next_observations, nextStateActionsContinuous)
                    minQNextTarget = torch.min(QFunction1NextTarget, QFunction2NextTarget) - alphaContinuous * nextStateLogProbs
                    nextQValue = torch.tensor(sampledExperiences.rewards) + torch.logical_not(torch.tensor(sampledExperiences.dones)) * gamma * (minQNextTarget).view(-1)
                
                    QFunction1ActionValues = QNet[behavior].QNet1(sampledExperiences.observations, sampledExperiences.actionsContinuous).view(-1)
                    QFunction2ActionValues = QNet[behavior].QNet2(sampledExperiences.observations, sampledExperiences.actionsContinuous).view(-1)

                elif agents[behavior].usingDiscreteActions:
                    _, nextStateLogProbs, nextStateActionProbs, _ = agents[behavior].getDiscreteActionAndValue(sampledExperiences.nextObservations)
                    QFunction1NextTarget = QNet[behavior].QNet1Target(sampledExperiences.nextObservations)
                    QFunction2NextTarget = QNet[behavior].QNet2Target(sampledExperiences.nextObservations)
                    minQNextTarget = nextStateActionProbs * (torch.min(QFunction1NextTarget, QFunction2NextTarget) - alphaDiscrete * nextStateLogProbs)
                    minQNextTarget = minQNextTarget.sum(dim=1)
                    nextQValue = torch.tensor(sampledExperiences.rewards) + torch.logical_not(torch.tensor(sampledExperiences.dones)) * gamma * (minQNextTarget)
            
                    QFunction1ActionValues = QNet[behavior].QNet1(sampledExperiences.observations).gather(1, sampledExperiences.actionsDiscrete).view(-1)
                    QFunction2ActionValues = QNet[behavior].QNet2(sampledExperiences.observations).gather(1, sampledExperiences.actionsDiscrete).view(-1)

            QFunction1Loss = F.mse_loss(QFunction1ActionValues, nextQValue)
            QFunction2Loss = F.mse_loss(QFunction2ActionValues, nextQValue)
            QFunctionsTotalLoss = QFunction1Loss + QFunction2Loss
            
            QNet[behavior].QNetsOptimizer.zero_grad()
            QFunctionsTotalLoss.backward()
            QNet[behavior].QNetsOptimizer.step()



            # Actor update
            nextStateActionsContinuous, nextStateLogprobsContinuous, _ = agents[behavior].getContinuousActionAndValue(sampledExperiences.nextObservations)
            nextStateActionsDiscrete, nextStateLogprobsDiscrete, nextStateProbsDiscrete, _ = agents[behavior].getDiscreteActionAndValue(sampledExperiences.nextObservations)
            QFunction1Evaluation = QNet[behavior].QNet1(sampledExperiences.nextObservations, nextStateActionsContinuous)
            QFunction2Evaluation = QNet[behavior].QNet2(sampledExperiences.nextObservations, nextStateActionsContinuous)

            minQEvaluation = torch.min(QFunction1Evaluation, QFunction2Evaluation)

            policyLossContinuous = ((alphaContinuous * nextStateLogprobsContinuous - minQEvaluation)).sum(1).mean()
            policyLossDiscrete = (nextStateProbsDiscrete * (alphaDiscrete * nextStateLogprobsDiscrete - minQEvaluation)).sum(1).mean()
            policyLoss = policyLossContinuous + policyLossDiscrete

            agents[behavior].policyOptimizer.zero_grad()
            policyLoss.backward()
            agents[behavior].policyOptimizer.step()
            
            # TODO: Auto entropy
            # with torch.no_grad():
            #     a_c, a_d, lpi_c, lpi_d, p_d = pg.get_action(s_obs, device)
            # alpha_loss = (-log_alpha * p_d * (p_d * lpi_c + target_entropy)).sum(1).mean()
            # alpha_d_loss = (-log_alpha_d * p_d * (lpi_d + target_entropy_d)).sum(1).mean()

            # a_optimizer.zero_grad()
            # alpha_loss.backward()
            # a_optimizer.step()
            # alpha = log_alpha.exp().detach().cpu().item()

            # a_d_optimizer.zero_grad()
            # alpha_d_loss.backward()
            # a_d_optimizer.step()
            # alpha_d = log_alpha_d.exp().detach().cpu().item()

            # update the target network
            if i % 2 == 0:
                for param, targetParam in zip(QNet[behavior].QNet1.parameters(), QNet[behavior].QNet1_target.parameters()):
                    targetParam.data.copy_(QNet[behavior].tau*param.data + (1 - QNet[behavior].tau)*targetParam.data)
                for param, targetParam in zip(QNet[behavior].QNet2.parameters(), QNet[behavior].QNet2_target.parameters()):
                    targetParam.data.copy_(QNet[behavior].tau*param.data + (1 - QNet[behavior].tau)*targetParam.data)
env.close()