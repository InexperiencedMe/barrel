import numpy as np
import torch
from mlagents_envs.environment import UnityEnvironment, ActionTuple
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from collections import deque, namedtuple
import random

class UnityInterface():
    def __init__(self, envName=None):
        self.env = UnityEnvironment(file_name=envName, side_channels=[])
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

    def getInitialObservations(self, bufferList):
        for behavior in self.behaviorNames:
            decisionSteps, _ = self.env.get_steps(behavior)
            for agentNr in decisionSteps:
                bufferList[agentNr] = decisionSteps[agentNr].obs
        print(f"Returning initial observations with {sum(obs is None for obs in bufferList)} None elements")
        return bufferList

def layerInit(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPO(nn.Module):
    # TODO: Make architecture flexible. Set sizes and numer of hidden layers in few lines
    def __init__(self, envSpecs):
        super(PPO, self).__init__()
        self.envSpecs = envSpecs
        self.obsSize1D, self.obsSize3D = self.getObsSizes(self.envSpecs)
        self.obsChannels3D = self.obsSize3D[0] # first shape dim is channels
        self.continuousActionSize = self.envSpecs["ContinuousActions"]
        self.preCritic3DoutputSize = 0
        self.preActor3DoutputSize = 0
        self.using1Dobs = self.obsSize1D > 0
        self.using3Dobs = sum(self.obsSize3D) > 0

        assert self.using1Dobs or self.using3Dobs, "No 1D or 3D observations and you expect it to work?!?!?!?"

        if self.using1Dobs:
            self.preCritic1D = nn.Sequential(
                layerInit(nn.Linear(self.obsSize1D, 256)), nn.Tanh(),
                layerInit(nn.Linear(256, 128)), nn.Tanh(),
                layerInit(nn.Linear(128, 64)), nn.Tanh())
            
            self.preActor1D = nn.Sequential(
                layerInit(nn.Linear(self.obsSize1D, 256)), nn.Tanh(),
                layerInit(nn.Linear(256, 128)), nn.Tanh(),
                layerInit(nn.Linear(128, 64)), nn.Tanh())
        
        if self.using3Dobs:
            self.preCritic3D = nn.Sequential(
                layerInit(nn.Conv2d(self.obsChannels3D, 32, 8, stride=4)), nn.Tanh(),
                layerInit(nn.Conv2d(32, 64, 4, stride=2)), nn.Tanh(),
                layerInit(nn.Conv2d(64, 64, 3, stride=1)), nn.Tanh(), nn.Flatten(start_dim=0)) # defaults to 1, we will maintain batch dim with time

            self.preActor3D = nn.Sequential(
                layerInit(nn.Conv2d(self.obsChannels3D, 32, 8, stride=4)), nn.Tanh(),
                layerInit(nn.Conv2d(32, 64, 4, stride=2)), nn.Tanh(),
                layerInit(nn.Conv2d(64, 64, 3, stride=1)), nn.Tanh(), nn.Flatten(start_dim=0))
            
            self.preCritic3DoutputSize = self.calculateConvNetOutputSize(self.preCritic3D, self.obsSize3D)
            self.preActor3DoutputSize = self.calculateConvNetOutputSize(self.preActor3D, self.obsSize3D)

        self.criticFinal = nn.Sequential(layerInit(nn.Linear(64 + self.preCritic3DoutputSize, 1), std=0.01))        

        self.actorContinuous = nn.Sequential(
            layerInit(nn.Linear(64 + self.preActor3DoutputSize, 64)), nn.Tanh(),
            layerInit(nn.Linear(64, self.continuousActionSize), std=0.01))        
        self.actorLogStd = nn.Parameter(torch.zeros(self.continuousActionSize))

        self.actorDiscrete = []
        for discreteActionSize in self.envSpecs["DiscreteActions"]:
            self.actorDiscrete.append(nn.Sequential(
            layerInit(nn.Linear(64 + self.preActor3DoutputSize, 64)), nn.Tanh(),
            layerInit(nn.Linear(64, discreteActionSize), std=0.01)))

    def evaluateState(self, x):
        obs1D, obs3D = self.processObservations(x)
        return self.criticFinal(self.getObservationFeaturesForCritic(obs1D, obs3D))
    
    # TODO: Should combine the 2 action types and return empty action if not needed
    def getDiscreteActionAndValue(self, x, action=None):
        obs1D, obs3D = self.processObservations(x)
        observationFeatures = self.getObservationFeaturesForActor(obs1D, obs3D)

        if action is None:
            actionTemp = []
        else:
            actionTemp = action
            
        logProbs = []
        entropies = []
        # TODO: not handling action masks yet
        for discreteAction in range(len(self.envSpecs["DiscreteActions"])):
            logits = self.actorDiscrete[discreteAction](observationFeatures)
            probabilities = Categorical(logits=logits)
            if action is None:
                actionTemp.append(probabilities.sample())
                logProbs.append(probabilities.log_prob(actionTemp[discreteAction]))
                logProbs.append(probabilities.entropy())
        # TODO: I'd like to break it down so it doesnt calculate logprobs when I need only actions
        return actionTemp, torch.tensor(logProbs).sum(-1), torch.tensor(entropies).sum(-1), self.evaluateState(x)
    
    def getContinuousActionAndValue(self, x, action=None, evaluation=False):
        obs1D, obs3D = self.processObservations(x)
        observationFeatures = self.getObservationFeaturesForActor(obs1D, obs3D)

        actionMean = self.actorContinuous(observationFeatures)
        actionLogStd = self.actorLogStd.expand_as(actionMean)
        actionStd = torch.exp(actionLogStd)
        probabilities = Normal(actionMean, actionStd)
        if action is None:
            if evaluation == True:
                action = actionMean
            else:
                action = probabilities.rsample()
        # TODO: I'd like to break it down so it doesnt calculate logprobs when I need only actions
        return action, probabilities.log_prob(action).sum(-1), probabilities.entropy().sum(-1), self.evaluateState(x)

    def processObservations(self, x):
        # TODO Getting a warning with this approach.. Could rework
        obs1D = torch.empty((0,))
        obs3D = torch.empty((0,))
        for observation in x:
            observation = torch.tensor(observation)
            if len(observation.shape) == 1:
                obs1D = torch.cat((obs1D, observation))
            elif len(observation.shape) == 3:
                obs3D = torch.cat((obs3D, observation))
            else:
                print(f"Unexpected {len(observation.shape)}-dimensional observation")
        # TODO: Put it on device?
        # print(f"processObservations returning obs1D of shape {obs1D.shape} abd obs3D of shape {obs3D.shape}")
        return obs1D, obs3D
    
    def getObsSizes(self, specs):
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
    
    def calculateConvNetOutputSize(self, net, inputSize):
        return torch.numel(torch.flatten(net(torch.ones(inputSize))))
    
    def getObservationFeaturesForCritic(self, obs1D, obs3D):
        netsOutputs = []
        if self.using1Dobs:
            netsOutputs.append(self.preCritic1D(obs1D))
        if self.using3Dobs:
            netsOutputs.append(self.preCritic3D(obs3D))
        return torch.cat(netsOutputs)

    def getObservationFeaturesForActor(self, obs1D, obs3D):
        netsOutputs = []
        if self.using1Dobs:
            # print(f"Trying to feed preActor1D an input of shape {obs1D.shape} while obsSize1D is {self.obsSize1D}")
            netsOutputs.append(self.preActor1D(obs1D))
        if self.using3Dobs:
            netsOutputs.append(self.preActor3D(obs3D))
        return torch.cat(netsOutputs)
    
class Memory(object):
    def __init__(self, capacity, fieldNames="observations, actions, rewards, dones, nextObservations"):
        self.fieldNames = namedtuple("fieldNames", fieldNames)
        self.memory = deque(maxlen=capacity)

    def push(self, observation, action, reward, done, nextObservation):
        self.memory.append(self.fieldNames(observation, action, reward, done, nextObservation))

    def sample(self, batchSize):
        sampledEntries = random.sample(self.memory, batchSize)
        return self.fieldNames(*zip(*sampledEntries))

    def __len__(self):
        return len(self.memory)
    