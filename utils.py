import numpy as np
import torch
from mlagents_envs.environment import UnityEnvironment, ActionTuple
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from collections import deque, namedtuple
import random
from itertools import product
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Make debugging modular. Functions should have print statements when DEBUG param is passed

class UnityInterface():
    def __init__(self, envName, seed):
        self.env = UnityEnvironment(file_name=envName, seed=seed)
        self.env.reset()
        self.behaviorNames = list(self.env.behavior_specs)
        self.specs = self.prepareSpecs()
    
    def prepareSpecs(self):
        specs = {}
        for behavior in self.behaviorNames:
            behaviorSpecs = {}
            obsSpecs = self.env.behavior_specs[behavior].observation_specs
            behaviorSpecs["AgentsCount"] = self.countAgents(behavior)
            behaviorSpecs["Observations"] = [obsSpecs[i].shape for i in range(len(obsSpecs))]
            behaviorSpecs["ContinuousActions"] = self.env.behavior_specs[behavior].action_spec.continuous_size
            behaviorSpecs["DiscreteActions"] = self.env.behavior_specs[behavior].action_spec.discrete_branches
            specs[behavior] = behaviorSpecs
        return specs
        
    def getSpecs(self, behaviorName=None):
        if behaviorName == None:
            return self.specs
        else:
            return self.specs[behaviorName]
    
    def getBehaviorNames(self):
        return self.behaviorNames
    
    def getSteps(self, behaviorName):
        decisionSteps, terminalSteps = self.env.get_steps(behaviorName)
        return decisionSteps, terminalSteps

    def setActions(self, behaviorName, continuousActions = None, discreteActions = None):
        self.env.set_actions(behaviorName, ActionTuple(continuous=continuousActions, discrete=discreteActions))

    def step(self):
        self.env.step()

    def reset(self):
        self.env.reset()

    def close(self):
        self.env.close()

    def countAgents(self, behaviorName=None):
        if behaviorName is None:
            agentsCount = 0
            for behavior in self.behaviorNames:
                decisionSteps, terminalSteps = self.getSteps(behaviorName)
                agentsCount += len(set(decisionSteps).union(set(terminalSteps)))
        else:
            decisionSteps, terminalSteps = self.getSteps(behaviorName)
            agentsCount = len(set(decisionSteps).union(set(terminalSteps)))
        return agentsCount

# gpt magic. Indexing (batchSize, 3, 3, 3, 2) evaluation tensor with actions (batchSize, 4)
# So I can take evaluation of the state if I took action (2, 1, 2, 0) for example.
# Works well on multidiscrete of any size, checked by hand
def gatherEvaluationOfTakenActions(tensor, index):
    index = index.to(tensor.device)
    batch_indices = [torch.arange(tensor.size(0), device=tensor.device)]
    dim_indices = [index[:, i] for i in range(index.size(1))]
    all_indices = batch_indices + dim_indices
    return tensor[tuple(all_indices)]

def calculateConvNetOutputSize(net, inputSize):
    return torch.numel(net(torch.ones(inputSize)))

def getObsSizes(specs):
    obsSize1D = 0
    osbSize3D = [0, 0, 0]
    for obsShape in specs["Observations"]:
        if len(obsShape) == 1:
            obsSize1D += obsShape[0]
        elif len(obsShape) == 3:
            for i, size in enumerate(obsShape):
                osbSize3D[i] += size
        else:
            print(f"Unexpected {len(obsShape)}-dimensional observation")
    return obsSize1D, osbSize3D

# In Unity they have observation components that we have to deal with.
# Observations come as tuple of ndarrays of either 1D or 3D, I simply concatenate it into one of each type
def processObservations(x):
    allObs1D, allObs3D = [], []
    for observation in x:
        obs1D, obs3D = [], []
        for observationElement in observation:
            observationElement = torch.tensor(observationElement)
            if observationElement.ndim == 1:
                obs1D.append(observationElement)
            elif observationElement.ndim == 3:
                obs3D.append(observationElement)
            else:
                print(f"Unexpected {observationElement.ndim}-dimensional observation")
        if obs1D:
            allObs1D.append(torch.cat(obs1D))
        if obs3D:
            allObs3D.append(torch.cat(obs3D))

    final1D = torch.stack(allObs1D).to(device) if allObs1D else torch.empty(0, dtype=torch.float32, device=device)
    final3D = torch.stack(allObs3D).to(device) if allObs3D else torch.empty(0, dtype=torch.float32, device=device)
    return final1D, final3D

class QNetwork(nn.Module):
    def __init__(self, envSpecs):
        super(QNetwork, self).__init__()
        self.envSpecs = envSpecs
        self.obsSize1D, self.obsSize3D = getObsSizes(self.envSpecs)
        self.obsChannels3D = self.obsSize3D[-3] # third last element is number of channels. F.e. 3x128x128
        self.continuousActionSize = self.envSpecs["ContinuousActions"]
        self.preCritic1DoutputSize = 0
        self.preCritic3DoutputSize = 0
        self.using1Dobs = self.obsSize1D > 0
        self.using3Dobs = sum(self.obsSize3D) > 0
        self.usingDiscreteActions = len(self.envSpecs["DiscreteActions"]) > 0
        self.usingContinuousActions = self.continuousActionSize > 0

        self.outputSize = np.array(self.envSpecs["DiscreteActions"]).prod() if self.usingDiscreteActions else 1

        if self.using1Dobs:
            self.preCritic1DoutputSize = 256
            self.preCritic1D = nn.Sequential(
                nn.Linear(self.obsSize1D, 512), nn.ReLU(),
                nn.Linear(512, self.preCritic1DoutputSize), nn.ReLU())
        
        if self.using3Dobs:
            self.preCritic3D = nn.Sequential(
                nn.Conv2d(self.obsChannels3D, 16, 7, stride=4), nn.ReLU(),
                nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(), nn.Flatten())
            with torch.no_grad():
                self.preCritic3DoutputSize = calculateConvNetOutputSize(self.preCritic3D, self.obsSize3D)

        self.criticFinal = nn.Sequential(nn.Linear(self.preCritic1DoutputSize + self.preCritic3DoutputSize + self.getTotalActionSize(), 128),
                                         nn.ReLU(), nn.Linear(128, self.outputSize))
    
    def forward(self, x, actionsContinuous=None, actionsDiscrete=None):
        obs1D, obs3D = processObservations(x)
        standardObsFeatures = self.getObservationFeaturesForCritic(obs1D, obs3D)
        actionObsFeatures = self.getActionRepresentationForInput(actionsContinuous, actionsDiscrete)
        out = self.criticFinal(torch.cat((standardObsFeatures, actionObsFeatures), -1))
        # print(f"IN CRITIC FORWARD We got obsF: {standardObsFeatures}, obsA: {actionObsFeatures} and outputting {out}")
        return out.reshape(-1, *self.envSpecs["DiscreteActions"]) if self.usingDiscreteActions else out.reshape(-1)
        
    def getObservationFeaturesForCritic(self, obs1D, obs3D):
        if self.using1Dobs and self.using3Dobs:
            return torch.cat((self.preCritic1D(obs1D), self.preCritic3D(obs3D)), -1)
        elif self.using1Dobs:
            return self.preCritic1D(obs1D)
        elif self.using3Dobs:
            return self.preCritic3D(obs3D)

    def getTotalActionSize(self):
        return self.continuousActionSize + sum(self.envSpecs["DiscreteActions"])
    
    def getActionRepresentationForInput(self, actionsContinuous=None, actionsDiscrete=None):
        featuresList = []
        if self.usingContinuousActions:
            featuresList.append(actionsContinuous)

        if self.usingDiscreteActions:
            onehots = []    
            for i, discreteSize in enumerate(self.envSpecs["DiscreteActions"]):
                onehots.append(F.one_hot(actionsDiscrete[:, i], discreteSize))
            featuresList.append(torch.cat(onehots, -1))
    
        features = torch.cat(featuresList, -1)
        return features

# class SoftQNetwork():
#     def __init__(self, envSpecs, criticLR):
#         self.QFunction1 = QNetwork(envSpecs).to(device)
#         self.QFunction2 = QNetwork(envSpecs).to(device)
#         self.QFunction1Target = QNetwork(envSpecs).to(device)
#         self.QFunction2Target = QNetwork(envSpecs).to(device)
#         self.QFunction1Target.load_state_dict(self.QFunction1.state_dict())
#         self.QFunction2Target.load_state_dict(self.QFunction2.state_dict())
#         self.criticOptimizer = optim.Adam(list(self.QFunction1.parameters()) + list(self.QFunction2.parameters()), lr=criticLR, eps=1e-4)
#         self.tau = 0.005

LOG_STD_MAX = 2
LOG_STD_MIN = -10

class SAC(nn.Module):
    # TODO: Make architecture flexible. Set sizes and numer of hidden layers in few lines
    # TODO: Think about what variable should be local. Not all vars need "self."
    def __init__(self, envSpecs, continuousActionLowBound = -1, continuousActionHighBound = 1):
        super(SAC, self).__init__()
        self.envSpecs = envSpecs
        self.obsSize1D, self.obsSize3D = getObsSizes(self.envSpecs)
        self.obsChannels3D = self.obsSize3D[0] # first shape dim is channels
        self.continuousActionSize = self.envSpecs["ContinuousActions"]
        self.preActor1DoutputSize = 0
        self.preActor3DoutputSize = 0
        self.using1Dobs = self.obsSize1D > 0
        self.using3Dobs = sum(self.obsSize3D) > 0
        self.usingDiscreteActions = len(self.envSpecs["DiscreteActions"]) > 0
        self.usingContinuousActions = self.continuousActionSize > 0
        assert self.using1Dobs or self.using3Dobs, "No 1D or 3D observations and you expect it to work?!?!?!?"
        assert self.usingDiscreteActions or self.usingContinuousActions, "We have to use either continuous or discrete actions"

        if self.using1Dobs:      
            self.preActor1DoutputSize = 256
            self.preActor1D = nn.Sequential(
                nn.Linear(self.obsSize1D, 512), nn.ReLU(),
                nn.Linear(512, self.preActor1DoutputSize), nn.ReLU())
            
        if self.using3Dobs:
            self.preActor3D = nn.Sequential(
                nn.Conv2d(self.obsChannels3D, 16, 7, stride=4), nn.ReLU(),
                nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(), nn.Flatten())
            self.preActor3DoutputSize = calculateConvNetOutputSize(self.preActor3D, self.obsSize3D)

        if self.usingContinuousActions:
            self.actorContinuous = nn.Sequential(
                nn.Linear(self.preActor1DoutputSize + self.preActor3DoutputSize, 128), nn.ReLU(),
                nn.Linear(128, self.continuousActionSize))        
            self.actorLogStd = nn.Linear(self.preActor1DoutputSize + self.preActor3DoutputSize, self.continuousActionSize)
            self.register_buffer("continuousActionScale", torch.tensor((continuousActionHighBound - continuousActionLowBound) / 2.0, dtype=torch.float32))
            self.register_buffer("continuousActionBias", torch.tensor((continuousActionHighBound + continuousActionLowBound) / 2.0, dtype=torch.float32))

        if self.usingDiscreteActions:
            self.actorDiscrete = nn.Sequential(
                nn.Linear(self.preActor1DoutputSize + self.preActor3DoutputSize, 128), nn.ReLU(),
                nn.Linear(128, sum(self.envSpecs["DiscreteActions"])))
            # print(f"So because we have actions defined as {self.envSpecs['DiscreteActions']}, are output discrete layer is of size {sum(self.envSpecs['DiscreteActions'])}")
        
    # TODO: not handling action masks yet
    # TODO: Should combine the 2 action types and return empty action if not needed
    def getDiscreteAction(self, x, action=None, withLogProbs=True):
        obs1D, obs3D = processObservations(x)
        observationFeatures = self.getObservationFeaturesForActor(obs1D, obs3D)
        unsplitLogits = self.actorDiscrete(observationFeatures)
        splitLogits = torch.split(unsplitLogits, list(self.envSpecs["DiscreteActions"]), dim=-1)
        actionDistributions = [Categorical(logits=logits) for logits in splitLogits]
        if action is None:
            action = torch.stack([distribution.sample() for distribution in actionDistributions])

        if withLogProbs:
            logProbsList = [F.log_softmax(logits, dim=-1) for logits in splitLogits]
            probsList = [distribution.probs for distribution in actionDistributions]
            # print(f"logProbsList:\n{logProbsList},\nprobsList:\n{probsList}")

            
            # calculating finalLogProbs and finalProbs for multidiscrete actions:
            # - take list of logprobs and distribution probs
            # - permute them all together column wise, so for multidiscrete of size (3, 3, 3, 2) we have 3*3*3*2=54 permutations
            # - Now we have logprobs and probs of any specific combination of actions. Logprobs summed, probs multiplied
            # It's hard to see on PushBlock env of discrete action (7). WallJump of action (3, 3, 3, 2) is when the approach truly shines

            columnsLogprobs = [[discreteActionLogprobs[:, i] for i in range(discreteActionLogprobs.shape[-1])] for discreteActionLogprobs in logProbsList]
            combinationsLogProbs = list(product(*columnsLogprobs))
            finalLogProbs = torch.stack([sum(combination) for combination in combinationsLogProbs], -1)
            # print(f"finalLogProbs:\n{finalLogProbs} of shape {finalLogProbs.shape}")
            finalLogProbs = finalLogProbs.reshape(-1, *self.envSpecs["DiscreteActions"])
            # print(f"these logprobs we reshape into:\n{finalLogProbs} of shape {finalLogProbs.shape}")
            

            columnsProbs = [[discreteActionProbs[:, i] for i in range(discreteActionProbs.shape[-1])] for discreteActionProbs in probsList]
            combinationsProbs = list(product(*columnsProbs))
            finalProbs = torch.stack([torch.prod(torch.stack(combination), 0) for combination in combinationsProbs], -1)
            # print(f"finalProbs:\n{finalProbs} of shape {finalProbs.shape}")
            finalProbs = finalProbs.reshape(-1, *self.envSpecs["DiscreteActions"])
            # print(f"these probs we reshape into:\n{finalProbs} of shape {finalProbs.shape}\n\n")
        else:
            finalLogProbs, finalProbs = None, None
        
        return action.T, finalLogProbs, finalProbs
        
    def getContinuousAction(self, x, evaluation=False, withLogProbs=True):
        obs1D, obs3D = processObservations(x)
        observationFeatures = self.getObservationFeaturesForActor(obs1D, obs3D)
        actionMean = self.actorContinuous(observationFeatures)
        actionLogStd = self.actorLogStd(observationFeatures)
        actionLogStd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (actionLogStd + 1)  # From SpinUp. Keeps bounds transforming range -1:1 to min:max
        actionStd = actionLogStd.exp()
        distribution = Normal(actionMean, actionStd)
        if evaluation == True:
            actionSample = actionMean
        else:
            actionSample = distribution.rsample()

        actionSampleTanh = torch.tanh(actionSample)
        action = actionSampleTanh * self.continuousActionScale + self.continuousActionBias
        
        if withLogProbs:
            logProbs = distribution.log_prob(actionSample)
            logProbs -= torch.log(self.continuousActionScale * (1 - actionSampleTanh.pow(2)) + 1e-6)#.sum(-1, keepdim=True) # CleanRL version
            logProbs = logProbs.sum(-1).view(-1)
            # print(f"returning continuous logprobs of shape {logProbs.shape}")
        else:
            logProbs = None
        return action, logProbs

    def getObservationFeaturesForActor(self, obs1D, obs3D):
        if self.using1Dobs and self.using3Dobs:
            return torch.cat((self.preActor1D(obs1D), self.preActor3D(obs3D)), -1)
        elif self.using1Dobs:
            output = self.preActor1D(obs1D)
            return output
        elif self.using3Dobs:
            return self.preActor3D(obs3D)
    
    
class Memory(object):
    def __init__(self, capacity, fieldNames=["observations", "actionsContinuous", "actionsDiscrete", "rewards", "dones", "nextObservations"]):
        self.fieldNames = namedtuple("fieldNames", fieldNames)
        self.memory = deque(maxlen=capacity)

    def push(self, observation, actionContinuous, actionDiscrete, reward, done, nextObservation):
        self.memory.append(self.fieldNames(observation, actionContinuous, actionDiscrete, reward, done, nextObservation))

    def sample(self, batchSize):
        sampledEntries = random.sample(self.memory, batchSize)
        a = self.fieldNames(*zip(*sampledEntries))
        return a

    def __len__(self):
        return len(self.memory)
    