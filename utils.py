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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Make debugging modular. Functions should have print statements when DEBUG param is passed

class UnityInterface():
    def __init__(self, envName=None):
        self.env = UnityEnvironment(file_name=envName)
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
            allObs3D.append(torch.cat(obs3D))  # Assuming concatenation along a suitable axis

    final1D = torch.stack(allObs1D).to(device) if allObs1D else torch.empty(0, device=device)
    final3D = torch.stack(allObs3D).to(device) if allObs3D else torch.empty(0, device=device)
    return final1D, final3D

class QNetwork(nn.Module):
    def __init__(self, envSpecs):
        super(QNetwork, self).__init__()
        self.envSpecs = envSpecs
        self.obsSize1D, self.obsSize3D = getObsSizes(self.envSpecs)
        self.obsChannels3D = self.obsSize3D[-3] # first shape dim is channels
        self.continuousActionSize = self.envSpecs["ContinuousActions"]
        self.preCritic1DoutputSize = 0
        self.preCritic3DoutputSize = 0
        self.using1Dobs = self.obsSize1D > 0
        self.using3Dobs = sum(self.obsSize3D) > 0
        self.usingDiscreteActions = len(self.envSpecs["DiscreteActions"]) > 0
        self.usingContinuousActions = self.continuousActionSize > 0

        if self.using1Dobs:
            self.preCritic1DoutputSize = 128
            self.preCritic1D = nn.Sequential(
                nn.Linear(self.obsSize1D, 256), nn.ReLU(),
                nn.Linear(256, self.preCritic1DoutputSize), nn.ReLU())
        
        if self.using3Dobs:
            self.preCritic3D = nn.Sequential(
                nn.Conv2d(self.obsChannels3D, 16, 7, stride=4), nn.ReLU(),
                nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(), nn.Flatten())
            self.preCritic3DoutputSize = calculateConvNetOutputSize(self.preCritic3D, self.obsSize3D)

        self.criticFinal = nn.Sequential(nn.Linear(self.preCritic1DoutputSize + self.preCritic3DoutputSize + self.getTotalActionSize(), 128), nn.ReLU(), nn.Linear(128, 1))
    
    def forward(self, x, actionsContinuous=None, actionsDiscrete=None):
        obs1D, obs3D = processObservations(x)
        standardObsFeatures = self.getObservationFeaturesForCritic(obs1D, obs3D)
        actionObsFeatures = self.getActionRepresentationForInput(actionsContinuous, actionsDiscrete)
        return self.criticFinal(torch.cat((standardObsFeatures, actionObsFeatures), -1))
        
    def getObservationFeaturesForCritic(self, obs1D, obs3D):
        if self.using1Dobs and self.using3Dobs:
            return torch.cat((self.preCritic1D(obs1D), self.preCritic3D(obs3D)), -1)
        elif self.using1Dobs:
            return self.preCritic1D(obs1D)
        elif self.using3Dobs:
            return self.preCritic3D(obs3D)
        
        # netsOutputs = []
        # if self.using1Dobs:
        #     netsOutputs.append(self.preCritic1D(obs1D))
        # if self.using3Dobs:
        #     netsOutputs.append(self.preCritic3D(obs3D))
        # return torch.cat(netsOutputs, -1)

    def getTotalActionSize(self):
        return self.continuousActionSize + sum(self.envSpecs["DiscreteActions"])
    
    def getActionRepresentationForInput(self, actionsContinuous=None, actionsDiscrete=None):
        continuousFeatures = torch.empty(0, device=device)
        discreteFeatures = torch.empty(0, device=device)
        
        if actionsContinuous != None:
            continuousFeatures = torch.cat((continuousFeatures, actionsContinuous), -1)
        if actionsDiscrete != None:
            discreteFeatures = torch.cat([F.one_hot(actionsDiscrete[:, i], size) for i, size in enumerate(self.envSpecs["DiscreteActions"])], -1)

        return torch.cat((continuousFeatures, discreteFeatures))

class SoftQNetwork():
    def __init__(self, envSpecs):
        self.QNet1 = QNetwork(envSpecs).to(device)
        self.QNet2 = QNetwork(envSpecs).to(device)
        self.QNet1Target = QNetwork(envSpecs).to(device)
        self.QNet2Target = QNetwork(envSpecs).to(device)
        self.QNet1Target.load_state_dict(self.QNet1.state_dict())
        self.QNet2Target.load_state_dict(self.QNet2.state_dict())

        self.QNetsOptimizer = optim.AdamW(list(self.QNet1.parameters()) + list(self.QNet2.parameters()), lr=1e-3)  
        self.tau = 0.005

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

        assert self.using1Dobs or self.using3Dobs, "No 1D or 3D observations and you expect it to work?!?!?!?"

        if self.using1Dobs:      
            self.preActor1DoutputSize = 128
            self.preActor1D = nn.Sequential(
                nn.Linear(self.obsSize1D, 128), nn.ReLU(),
                nn.Linear(128, self.preActor1DoutputSize), nn.ReLU())
            
        if self.using3Dobs:
            self.preActor3D = nn.Sequential(
                nn.Conv2d(self.obsChannels3D, 16, 7, stride=4), nn.ReLU(),
                nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(), nn.Flatten())
            self.preActor3DoutputSize = calculateConvNetOutputSize(self.preActor3D, self.obsSize3D)


        self.usingDiscreteActions = len(self.envSpecs["DiscreteActions"]) > 0
        self.usingContinuousActions = self.continuousActionSize > 0
        assert self.usingDiscreteActions or self.usingContinuousActions, "We have to use either continuous or discrete actions"

        if self.usingContinuousActions:
            self.actorContinuous = nn.Sequential(
                nn.Linear(self.preActor1DoutputSize + self.preActor3DoutputSize, 128), nn.ReLU(),
                nn.Linear(128, self.continuousActionSize))        
            self.actorLogStd = nn.Linear(self.preActor1DoutputSize + self.preActor3DoutputSize, self.continuousActionSize)
            self.register_buffer("continuousActionScale", torch.tensor((continuousActionHighBound - continuousActionLowBound) / 2.0, dtype=torch.float32))
            self.register_buffer("continuousActionBias", torch.tensor((continuousActionHighBound + continuousActionLowBound) / 2.0, dtype=torch.float32))


        if self.usingDiscreteActions:
            self.actorDiscrete = nn.Sequential(
                nn.Linear(self.preActor1DoutputSize + self.preActor3DoutputSize, 256), nn.ReLU(),
                nn.Linear(256, sum(self.envSpecs["DiscreteActions"])))
            # print(f"So because we have actions defined as {self.envSpecs['DiscreteActions']}, are output discrete layer is of size {sum(self.envSpecs['DiscreteActions'])}")
        
        self.actorOptimizer = optim.AdamW(list(self.parameters()), lr=3e-4)

    def forward(self, x):
        obs1D, obs3D = processObservations(x)
        observationFeatures = self.getObservationFeaturesForActor(obs1D, obs3D)
        actionSample = self.actorContinuous(observationFeatures)
        actionSampleTanh = torch.tanh(actionSample)
        action = actionSampleTanh * self.continuousActionScale + self.continuousActionBias
        return action

    # TODO: not handling action masks yet
    # TODO: Should combine the 2 action types and return empty action if not needed
    def getDiscreteActionAndValue(self, x, action=None):
        obs1D, obs3D = processObservations(x)
        observationFeatures = self.getObservationFeaturesForActor(obs1D, obs3D)
        logits = self.actorDiscrete(observationFeatures)
        split_logits = torch.split(logits, list(self.envSpecs["DiscreteActions"]), dim=-1)
        multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
        if action is None:
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        logprobs = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])     
        return action.T, logprobs.sum(0)
        
    def getContinuousActionAndValue(self, x, evaluation=False, withLogprobs=True):
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
        
        if withLogprobs:
            logProbs = distribution.log_prob(actionSample)
            logProbs -= torch.log(self.continuousActionScale * (1 - actionSampleTanh.pow(2)) + 1e-6)#.sum(-1, keepdim=True) # CleanRL version
            logProbs = logProbs.sum(-1).view(-1)
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
    