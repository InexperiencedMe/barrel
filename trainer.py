import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import *
# torch.set_printoptions(linewidth=100, precision=4, sci_mode=False, threshold=2000)

seed: int = 1
torch_deterministic: bool = True
totalTimesteps: int = 220000
bufferSize: int = int(1e4)
gamma: float = 0.98
tau: float = 0.005
batchSize: int = 128
learningStart: int = 2000
actorLR: float = 3e-4
criticLR: float = 3e-4
optimizationInterval: int = 1
stepInterval: int = 4
softCriticUpdateInterval: int = 1
targetEntropy_scale: float = 0.9
rewardScaling: float = 10
lossesPlotAveraging = 5
rewardsPlotAveraging = 1
graph = True
resume = True
saveCheckpoints = True
checkpointInterval: int = 10000
checkpointToLoad = f"checkpoints\\PushBlock-HOPE-880000.pth"
checkpointIDsubstring = "HOPE"

def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = torch_deterministic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# env = UnityInterface("Builds\\Windows\\Ball3D\\UnityEnvironment", seed=seed)              # 1D obs only, continuous action of size 2. Rewards: 0.1 for every step, -1 for fail, 100 is the max episodic return
# env = UnityInterface("Builds\\Windows\\Crawler\\UnityEnvironment", seed=seed)             # 1D obs only, continuous action of size 8
# env = UnityInterface("Builds\\Windows\\PushBlock\\UnityEnvironment", seed=seed)           # 1D obs only, discrete action of size (7). Rewards: 5 for win, -0.001 for every step
env = UnityInterface("Builds\\Windows\\PushBlockMov\\UnityEnvironment", seed=seed)           # 1D obs only, discrete action of size (7). Rewards: 5 for win, -0.001 for every step, mod with roughly 0.001 while moving the block
# env = UnityInterface("Builds\\Windows\\WallJump\\UnityEnvironment", seed=seed)            # 1D obs only, discrete action of size (3, 3, 3, 2)	# env = UnityInterface("Builds\\Windows\\WallJump\\UnityEnvironment", seed=seed)
# env = UnityInterface("Builds\\Windows\\Worm\\UnityEnvironment", seed=seed)

# env = UnityInterface("Builds/Linux/Ball3D/Ball3D", seed=seed)                             # 1D obs only, continuous action of size 2. Rewards: 0.1 for every step, -1 for fail, 100 is the max episodic return
# env = UnityInterface("Builds/Linux/PushBlock/PushBlock", seed=seed)                       # 1D obs only, discrete action of size (7). Rewards: 5 for win, -0.001 for every step
# env = UnityInterface("Builds/Linux/WallJump/WallJump", seed=seed)                         # 1D obs only, discrete action of size (3, 3, 3, 2)	# env = UnityInterface("Builds/Linux/WallJump/WallJump", seed=seed)

# env = UnityInterface(None, seed=seed)

print(f"{env.getSpecs()}")
behaviorNames = env.getBehaviorNames()

totalAgentsCounts = 0
for behavior in behaviorNames:
    totalAgentsCounts += (env.getSpecs(behavior)["AgentsCount"])

actor, actorOptimizer, memory, observationBuffer, actionsBuffer, = {}, {}, {}, {}, {}
QFunction1, QFunction2, QFunction1Target, QFunction2Target, criticOptimizer = {}, {}, {}, {}, {}
targetEntropyC, logAlphaC, alphaC, alphaOptimizerC = {}, {}, {}, {}
targetEntropyD, logAlphaD, alphaD, alphaOptimizerD = {}, {}, {}, {}
for behavior in behaviorNames:
    envSpecs = env.getSpecs(behavior)
    actor[behavior] = SAC(envSpecs).to(device)
    actorOptimizer[behavior] = optim.Adam(list(actor[behavior].parameters()), lr=actorLR, eps=1e-5)
    QFunction1[behavior] = QNetwork(envSpecs).to(device)
    QFunction2[behavior] = QNetwork(envSpecs).to(device)
    QFunction1Target[behavior] = QNetwork(envSpecs).to(device)
    QFunction2Target[behavior] = QNetwork(envSpecs).to(device)
    QFunction1Target[behavior].load_state_dict(QFunction1[behavior].state_dict())
    QFunction2Target[behavior].load_state_dict(QFunction2[behavior].state_dict())
    criticOptimizer[behavior] = optim.Adam(list(QFunction1[behavior].parameters()) + list(QFunction2[behavior].parameters()), lr=criticLR, eps=1e-5)
    
    memory[behavior] = Memory(bufferSize)
    assert actor[behavior].usingContinuousActions or actor[behavior].usingDiscreteActions, "Agent not using continuous nor discrete actions, VERY BAD"

    # Alpha tuning init
    alphaC[behavior] = torch.tensor((0), dtype=torch.float, device=device)
    if actor[behavior].usingContinuousActions:
        targetEntropyC[behavior] = -targetEntropy_scale * torch.tensor(env.getSpecs(behavior)["ContinuousActions"]).to(device)
        logAlphaC[behavior] = torch.zeros(1, requires_grad=True, device=device)
        alphaC[behavior] = logAlphaC[behavior].exp().item()
        alphaOptimizerC[behavior] = optim.Adam([logAlphaC[behavior]], lr=criticLR, eps=1e-5)
        
    alphaD[behavior] = torch.tensor((0), dtype=torch.float, device=device)
    if actor[behavior].usingDiscreteActions:
        targetEntropyD[behavior] = -targetEntropy_scale * torch.log(1 / torch.tensor(env.getSpecs(behavior)["DiscreteActions"]).prod().to(device))
        logAlphaD[behavior] = torch.zeros(1, requires_grad=True, device=device)
        alphaD[behavior] = logAlphaD[behavior].exp().item()
        alphaOptimizerD[behavior] = optim.Adam([logAlphaD[behavior]], lr=criticLR, eps=1e-5)

    if resume:
        checkpoint = torch.load(checkpointToLoad)
        actor[behavior].load_state_dict(checkpoint["actor"])
        actorOptimizer[behavior].load_state_dict(checkpoint["actorOptimizer"])
        QFunction1[behavior].load_state_dict(checkpoint["QFunction1"])
        QFunction2[behavior].load_state_dict(checkpoint["QFunction2"])
        QFunction1Target[behavior].load_state_dict(checkpoint["QFunction1Target"])
        QFunction2Target[behavior].load_state_dict(checkpoint["QFunction2Target"])
        criticOptimizer[behavior].load_state_dict(checkpoint["criticOptimizer"])
        if actor[behavior].usingContinuousActions:
            logAlphaC[behavior] = checkpoint["logAlphaC"]
            alphaC[behavior] = checkpoint["alphaC"]
            alphaOptimizerC[behavior].load_state_dict(checkpoint["alphaOptimizerC"])
        if actor[behavior].usingDiscreteActions:
            logAlphaD[behavior] = checkpoint["logAlphaD"]
            alphaD[behavior] = checkpoint["alphaD"]
            alphaOptimizerD[behavior].load_state_dict(checkpoint["alphaOptimizerD"])

# FIXME: Add actions buffer continuous and discrete and test 3DBall
observationBuffer = [None] * totalAgentsCounts
for i in range(totalAgentsCounts):
    actionsBuffer[i] = {'continuous': None, 'discrete': None}
rewards = np.zeros(totalAgentsCounts)

finalRewards, criticLosses, actorLosses, alphaLosses, alphasC, alphasD, QEvaluations, logProbs = [], [], [], [], [], [], [], []
start = checkpoint["globalStep"] + 1 if resume else 1
for globalStep in range(start - learningStart, start + totalTimesteps):
    if globalStep % stepInterval == 0 or globalStep < start:
        for behavior in behaviorNames:
            decisionSteps, terminalSteps = env.getSteps(behavior)
            observationsThatNeedAction = []
            for agent in decisionSteps:
                observation = decisionSteps[agent].obs
                observationsThatNeedAction.append(observation)
                reward = decisionSteps[agent].reward * rewardScaling
                lastObservation = observationBuffer[agent]
                lastActionContinuous = actionsBuffer[agent]['continuous']
                lastActionDiscrete = actionsBuffer[agent]['discrete']
                if lastObservation != None and (lastActionContinuous != None or lastActionDiscrete != None) and agent not in terminalSteps:
                    memory[behavior].push(lastObservation, lastActionContinuous, lastActionDiscrete, reward, False, observation)
                observationBuffer[agent] = observation
                rewards[agent] += reward
                if reward > 3:
                    print(f"D agent {agent} scored a goal. final reward: {rewards[agent]}")
                
            for agent in terminalSteps:
                observation = terminalSteps[agent].obs
                reward = terminalSteps[agent].reward * rewardScaling
                lastObservation = observationBuffer[agent]
                lastActionContinuous = actionsBuffer[agent]['continuous']
                lastActionDiscrete = actionsBuffer[agent]['discrete']
                # Technically could skip the action None check. If lastObs exist, action does too
                if lastObservation != None and (lastActionContinuous != None or lastActionDiscrete != None):
                    memory[behavior].push(lastObservation, lastActionContinuous, lastActionDiscrete, reward, True, observation)
                    observationBuffer[agent] = None
                    actionsBuffer[agent]['continuous'] = None
                    actionsBuffer[agent]['discrete'] = None
                    rewards[agent] += reward
                finalRewards.append(rewards[agent])
                rewards[agent] = 0

            
            behaviorActionsForThisStep = {}
            specs = env.getSpecs(behavior)
            nrOfContinuousActions = specs["ContinuousActions"] # TODO: Substitute it with actors[behavior].usingContinuousActions
            nrOfDiscreteActions = len(specs["DiscreteActions"])
            behaviorActionsForThisStep["continuous"] = torch.zeros((len(decisionSteps), nrOfContinuousActions), requires_grad=False, dtype=torch.float32, device=device)
            behaviorActionsForThisStep["discrete"] = torch.zeros((len(decisionSteps), nrOfDiscreteActions), requires_grad=False, dtype=torch.long, device=device)
            # print(f"Allocated discrete action buffer of shape {behaviorActionsForThisStep['discrete'].shape}")

            # Batched pass to get actions  
            if len(observationsThatNeedAction) > 0:
                if actor[behavior].usingContinuousActions:
                    behaviorActionsForThisStep["continuous"], _ = actor[behavior].getContinuousAction(observationsThatNeedAction, withLogProbs=False)
                if actor[behavior].usingDiscreteActions:
                    behaviorActionsForThisStep["discrete"], _, _ = actor[behavior].getDiscreteAction((observationsThatNeedAction), withLogProbs=False)

            # Transcribe the actions to buffer
                for j, agent in enumerate(decisionSteps):
                    if nrOfContinuousActions > 0:
                        actionsBuffer[agent]['continuous'] = behaviorActionsForThisStep['continuous'][j]
                    if nrOfDiscreteActions > 0:
                        actionsBuffer[agent]['discrete'] = behaviorActionsForThisStep['discrete'][j]

            env.setActions(behavior, behaviorActionsForThisStep['continuous'].detach().cpu().numpy(), behaviorActionsForThisStep['discrete'].detach().cpu().numpy())
        env.step()
    
    # ALGO LOGIC: training.
    if globalStep > start:
        if globalStep % optimizationInterval == 0:
            data = memory[behavior].sample(batchSize)
            observationsBatch       =               data.observations
            actionsContinuousBatch  =               torch.stack(data.actionsContinuous) if actor[behavior].usingContinuousActions else None
            actionsDiscreteBatch    =               torch.stack(data.actionsDiscrete) if actor[behavior].usingDiscreteActions else None
            rewardsBatch            =               torch.from_numpy(np.stack(data.rewards, dtype=np.float32)).to(device)
            isThereNextStepBatch    =               torch.logical_not(torch.tensor(np.stack(data.dones), device=device, dtype=torch.float32))
            nextObservationsBatch   =               data.nextObservations
            
            # #################### CRITIC UPDATE
            with torch.no_grad():
                nextStateActionsContinuous, nextStateActionsDiscrete, nextStateLogProbsContinuous, nextStateLogProbsDiscrete, nextStateProbsDiscrete, alphaTerm = None, None, 0, 0, 1, torch.ones(1, device=device, dtype=torch.float32)
                if actor[behavior].usingContinuousActions:
                    nextStateActionsContinuous, nextStateLogProbsContinuous = actor[behavior].getContinuousAction(nextObservationsBatch)
                if actor[behavior].usingDiscreteActions:
                    nextStateActionsDiscrete, nextStateLogProbsDiscrete, nextStateProbsDiscrete = actor[behavior].getDiscreteAction(nextObservationsBatch)

                QFunction1NextTarget = QFunction1Target[behavior](nextObservationsBatch, nextStateActionsContinuous, nextStateActionsDiscrete)
                QFunction2NextTarget = QFunction2Target[behavior](nextObservationsBatch, nextStateActionsContinuous, nextStateActionsDiscrete)
                minQNextTarget = nextStateProbsDiscrete * (torch.min(QFunction1NextTarget, QFunction2NextTarget) - (alphaC[behavior]*nextStateLogProbsContinuous + alphaD[behavior]*nextStateLogProbsDiscrete))
                if minQNextTarget.ndim > 1:
                    minQNextTarget = torch.sum(minQNextTarget, axis=tuple(range(1, minQNextTarget.ndim)))
                nextQValue = rewardsBatch + isThereNextStepBatch * gamma * minQNextTarget
                
            QFunction1ActionValues = QFunction1[behavior](observationsBatch, actionsContinuousBatch.detach() if actor[behavior].usingContinuousActions else None, actionsDiscreteBatch.detach() if actor[behavior].usingDiscreteActions else None)
            QFunction2ActionValues = QFunction2[behavior](observationsBatch, actionsContinuousBatch.detach() if actor[behavior].usingContinuousActions else None, actionsDiscreteBatch.detach() if actor[behavior].usingDiscreteActions else None)
            if actor[behavior].usingDiscreteActions:
                QFunction1ActionValues = gatherEvaluationOfTakenActions(QFunction1ActionValues, actionsDiscreteBatch)
                QFunction2ActionValues = gatherEvaluationOfTakenActions(QFunction2ActionValues, actionsDiscreteBatch)
            QFunction1Loss = F.mse_loss(QFunction1ActionValues, nextQValue)
            QFunction2Loss = F.mse_loss(QFunction2ActionValues, nextQValue)
            criticLoss = QFunction1Loss + QFunction2Loss
            criticOptimizer[behavior].zero_grad()
            criticLoss.backward()
            torch.nn.utils.clip_grad_norm_(QFunction1[behavior].parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(QFunction2[behavior].parameters(), max_norm=1.0)
            criticOptimizer[behavior].step()



            # #################### ACTOR UPDATE
            stateActionsContinuous, stateActionsDiscrete, stateLogProbsContinuous, stateLogProbsDiscrete, stateProbsDiscrete = None, None, torch.tensor(0, device=device), torch.tensor(0, device=device), torch.tensor(1, device=device)
            if actor[behavior].usingContinuousActions:
                stateActionsContinuous, stateLogProbsContinuous = actor[behavior].getContinuousAction(observationsBatch)
            if actor[behavior].usingDiscreteActions:
                stateActionsDiscrete, stateLogProbsDiscrete, stateProbsDiscrete = actor[behavior].getDiscreteAction(observationsBatch)

            QFunction1Evaluation = QFunction1[behavior](observationsBatch, stateActionsContinuous, stateActionsDiscrete)
            QFunction2Evaluation = QFunction1[behavior](observationsBatch, stateActionsContinuous, stateActionsDiscrete)
            minQEvaluation = torch.min(QFunction1Evaluation, QFunction2Evaluation)
            actorLoss = (stateProbsDiscrete * ((alphaC[behavior]*stateLogProbsContinuous + alphaD[behavior]*stateLogProbsDiscrete) - minQEvaluation)).mean()
            actorOptimizer[behavior].zero_grad()
            actorLoss.backward()
            torch.nn.utils.clip_grad_norm_(actor[behavior].parameters(), max_norm=1.0)
            actorOptimizer[behavior].step()



            # #################### ALPHA UPDATE
            if actor[behavior].usingContinuousActions:
                alphaLossC = (-logAlphaC[behavior].exp()*((stateLogProbsContinuous.to(device) + targetEntropyC[behavior]).detach())).mean()
                alphaOptimizerC[behavior].zero_grad()
                alphaLossC.backward()
                alphaOptimizerC[behavior].step()
                alphaC[behavior] = logAlphaC[behavior].exp().item()

            if actor[behavior].usingDiscreteActions:
                alphaLossD = (stateProbsDiscrete.detach()*(-logAlphaD[behavior].exp()*(stateLogProbsDiscrete.to(device) + targetEntropyD[behavior]).detach())).mean()
                alphaOptimizerD[behavior].zero_grad()
                alphaLossD.backward()
                alphaOptimizerD[behavior].step()
                alphaD[behavior] = logAlphaD[behavior].exp().item()



            # update the target networks
            if globalStep % softCriticUpdateInterval == 0:
                for param, targetParam in zip(QFunction1[behavior].parameters(), QFunction1Target[behavior].parameters()):
                    targetParam.data.copy_(tau * param.data + (1 - tau) * targetParam.data)
                for param, targetParam in zip(QFunction2[behavior].parameters(), QFunction2Target[behavior].parameters()):
                    targetParam.data.copy_(tau * param.data + (1 - tau) * targetParam.data)

            if globalStep % 1000 == 0:
                print(f"Step {behavior[:behavior.find('?')]} {globalStep:8}, Actor loss: {actorLoss:>12.4f}, QF loss: {criticLoss:>12.4f}")

            if globalStep % 10 == 0:
                criticLosses.append(criticLoss)
                actorLosses.append(actorLoss)
                alphasC.append(alphaC[behavior])
                alphasD.append(alphaD[behavior])
                QEvaluations.append(minQEvaluation.mean())
                logProbs.append((stateProbsDiscrete * (stateLogProbsContinuous + stateLogProbsDiscrete)).mean())
                
            if globalStep % checkpointInterval == 0 and saveCheckpoints:
                for behavior in behaviorNames:
                    checkpoint = {
                        'actor': actor[behavior].state_dict(),
                        'actorOptimizer': actorOptimizer[behavior].state_dict(),
                        'QFunction1': QFunction1[behavior].state_dict(),
                        'QFunction2': QFunction2[behavior].state_dict(),
                        'QFunction1Target': QFunction1Target[behavior].state_dict(),
                        'QFunction2Target': QFunction2Target[behavior].state_dict(),
                        'criticOptimizer': criticOptimizer[behavior].state_dict(),
                        'globalStep': globalStep,
                    }
                    if actor[behavior].usingContinuousActions:
                        checkpoint["logAlphaC"] = logAlphaC[behavior]
                        checkpoint["alphaC"] = alphaC[behavior]
                        checkpoint["alphaOptimizerC"] = alphaOptimizerC[behavior].state_dict()
                    if actor[behavior].usingDiscreteActions:
                        checkpoint["logAlphaD"] = logAlphaD[behavior]
                        checkpoint["alphaD"] = alphaD[behavior]
                        checkpoint["alphaOptimizerD"] = alphaOptimizerD[behavior].state_dict()

                    torch.save(checkpoint, f"checkpoints\\{behavior[:behavior.find('?')]}-{checkpointIDsubstring}-{globalStep}.pth")

                    if graph:
                        beginning = 0
                        dif = ((len(criticLosses) - beginning) % lossesPlotAveraging) + lossesPlotAveraging

                        # Style
                        plt.style.use('seaborn-v0_8-bright')

                        fig, (ax1, ax2) = plt.subplots(2, 1)
                        fig.set_size_inches(25.6, 14.4)

                        ax1.plot(torch.tensor(actorLosses[beginning:-dif]).view(-1, lossesPlotAveraging).mean(-1), label="actor loss")
                        ax1.plot(torch.tensor(alphaLosses[beginning:-dif]).view(-1, lossesPlotAveraging).mean(-1), label="alpha loss")
                        ax1.plot(torch.tensor(alphasC[beginning:-dif]).view(-1, lossesPlotAveraging).mean(-1), label="alphaC")
                        ax1.plot(torch.tensor(alphasD[beginning:-dif]).view(-1, lossesPlotAveraging).mean(-1), label="alphaD")
                        ax1.plot(torch.tensor(QEvaluations[beginning:-dif]).view(-1, lossesPlotAveraging).mean(-1), label="batch evaluation")
                        ax1.plot(torch.tensor(logProbs[beginning:-dif]).view(-1, lossesPlotAveraging).mean(-1), label="logprobs")
                        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fancybox=True, shadow=True).set_zorder(2)
                        ax1b = ax1.twinx()
                        ax1b.plot(torch.tensor(criticLosses[beginning:-dif]).view(-1, lossesPlotAveraging).mean(-1), 'm-', label="critic loss")
                        ax1b.set_ylabel("Critic Loss Value", color='m')
                        ax1b.tick_params(axis='y')
                        plt.gca().set_yscale('log')

                        ax1.set_title(f"{behavior[:behavior.find('?')]} {checkpointIDsubstring} {start-1} - {globalStep}")
                        ax1.set_xlabel(f"Iterations / {lossesPlotAveraging*10}")
                        ax1.set_ylabel(f"Value (avg of {lossesPlotAveraging})")
                        ax1.grid(True, linestyle='--', alpha=0.5)


                        dif_rewards = ((len(finalRewards) - beginning) % rewardsPlotAveraging) + rewardsPlotAveraging
                        ax2.plot(torch.tensor(finalRewards[beginning:-dif_rewards]).view(-1, rewardsPlotAveraging).mean(-1))
                        ax2.set_title("Final Rewards")
                        ax2.set_xlabel(f"Episode")
                        ax2.set_ylabel(f"Reward (avg of {rewardsPlotAveraging})")
                        
                        ax2.grid(True, linestyle='--', alpha=0.5)

                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(f"graphs\\{behavior[:behavior.find('?')]}-{checkpointIDsubstring}-{globalStep}", bbox_inches='tight')
                        plt.close('all')
                
env.close()

if graph:
    beginning = 0
    dif = ((len(criticLosses) - beginning) % lossesPlotAveraging) + lossesPlotAveraging

    # Style
    plt.style.use('seaborn-v0_8-bright')

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_size_inches(25.6, 14.4)

    ax1.plot(torch.tensor(actorLosses[beginning:-dif]).view(-1, lossesPlotAveraging).mean(-1), label="actor loss")
    ax1.plot(torch.tensor(alphaLosses[beginning:-dif]).view(-1, lossesPlotAveraging).mean(-1), label="alpha loss")
    ax1.plot(torch.tensor(alphasC[beginning:-dif]).view(-1, lossesPlotAveraging).mean(-1), label="alphaC")
    ax1.plot(torch.tensor(alphasD[beginning:-dif]).view(-1, lossesPlotAveraging).mean(-1), label="alphaD")
    ax1.plot(torch.tensor(QEvaluations[beginning:-dif]).view(-1, lossesPlotAveraging).mean(-1), label="batch evaluation")
    ax1.plot(torch.tensor(logProbs[beginning:-dif]).view(-1, lossesPlotAveraging).mean(-1), label="logprobs")
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fancybox=True, shadow=True).set_zorder(2)
    ax1b = ax1.twinx()
    ax1b.plot(torch.tensor(criticLosses[beginning:-dif]).view(-1, lossesPlotAveraging).mean(-1), 'm-', label="critic loss")
    ax1b.set_ylabel("Critic Loss Value", color='m')
    ax1b.tick_params(axis='y')
    plt.gca().set_yscale('log')

    ax1.set_title(f"{behavior[:behavior.find('?')]} {checkpointIDsubstring} {start-1} - {globalStep}")
    ax1.set_xlabel(f"Iterations / {lossesPlotAveraging}")
    ax1.set_ylabel(f"Value (avg of {lossesPlotAveraging})")
    ax1.grid(True, linestyle='--', alpha=0.5)


    dif_rewards = ((len(finalRewards) - beginning) % rewardsPlotAveraging) + rewardsPlotAveraging
    ax2.plot(torch.tensor(finalRewards[beginning:-dif_rewards]).view(-1, rewardsPlotAveraging).mean(-1))
    ax2.set_title("Final Rewards")
    ax2.set_xlabel(f"Episode")
    ax2.set_ylabel(f"Reward (avg of {rewardsPlotAveraging})")
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(left=0.04, right=0.96, bottom=0.056, hspace=0.23)
    # plt.subplot_tool()
    plt.show()