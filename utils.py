import numpy as np
import torch
from mlagents_envs.environment import UnityEnvironment, ActionTuple
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

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

    def countAgents(self, behaviorName):
        decisionSteps, _ = self.getSteps(behaviorName)
        return len(list(decisionSteps))
    
    def initializeMemory(self):
        pass

    def addExperiences(self):
        pass

def linearInitialize(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPO(nn.Module):
    def __init__(self, envSpecs):
        super(PPO, self).__init__()
        self.preCritic1D = nn.Sequential(
            linearInitialize(nn.Linear(obsSize1D, 256)), nn.Tanh(),
            linearInitialize(nn.Linear(256, 128)), nn.Tanh(),
            linearInitialize(nn.Linear(128, 64)), nn.Tanh())
        
        # TODO: Check if 3D obs exist
        self.preCritic3D = nn.Sequential(
            linearInitialize(nn.Conv2d(obs3Dchannels, 32, 8, stride=4)), nn.Tanh(),
            linearInitialize(nn.Conv2d(32, 64, 4, stride=2)), nn.Tanh(),
            linearInitialize(nn.Conv2d(64, 64, 3, stride=1)), nn.Tanh(), nn.Flatten())

        self.criticFinal = nn.Sequential(linearInitialize(nn.Linear(64 + preCritic3DoutputSize, 1), std=0.01))


        self.preActor1D = nn.Sequential(
            linearInitialize(nn.Linear(obsSize1D, 256)), nn.Tanh(),
            linearInitialize(nn.Linear(256, 128)), nn.Tanh(),
            linearInitialize(nn.Linear(128, 64)), nn.Tanh())
        
        self.preActor3D = nn.Sequential(
            linearInitialize(nn.Conv2d(obs3Dchannels, 32, 8, stride=4)), nn.Tanh(),
            linearInitialize(nn.Conv2d(32, 64, 4, stride=2)), nn.Tanh(),
            linearInitialize(nn.Conv2d(64, 64, 3, stride=1)), nn.Tanh(), nn.Flatten())
            
        self.actorContinuous = nn.Sequential(
            linearInitialize(nn.Linear(64 + preActor3DoutputSize, 64)), nn.Tanh(),
            linearInitialize(nn.Linear(64, continuousActionSize), std=0.01))
        self.actorLogStd = nn.Parameter(torch.zeros(continuousActionSize))

        self.actorDiscrete = []
        for discreteActionSize in discreteBranches:
            self.actorDiscrete.append(nn.Sequential(
            linearInitialize(nn.Linear(64, 64), nn.Tanh(),
            linearInitialize(nn.Linear(64, discreteActionSize), std=0.01))))

    def evaluateState(self, x):
        obs1D, obs3D = self.processObservations(x)
        return self.criticFinal(torch.cat((self.preCritic1D(obs1D), self.preCritic3D(obs3D))))
    
    def getDiscreteActionAndValue(self, x, action=None):
        obs1D, obs3D = self.processObservations(x)
        observationFeatures = torch.cat((self.preActor1D(obs1D), self.preActor3D(obs3D)))

        if action is None:
            actionTemp = []
        else:
            actionTemp = action
            
        logProbs = []
        entropies = []
        for discreteAction in range(discreteActionSize):
            logits = self.actorDiscrete(observationFeatures)
            probabilities = Categorical(logits=logits)
            if action is None:
                actionTemp.append(probabilities.sample())
                logProbs.append(probabilities.log_prob(actionTemp[discreteAction]))
                logProbs.append(probabilities.entropy())
        return actionTemp, torch.tensor(logProbs).sum(-1), torch.tensor(entropies).sum(-1), self.evaluateState(x)
    
    def getContinuousActionAndValue(self, x, action=None, evaluation=False):
        obs1D, obs3D = self.processObservations(x)
        observationFeatures = torch.cat((self.preActor1D(obs1D), self.preActor3D(obs3D)))

        actionMean = self.actorContinuous(observationFeatures)
        actionLogStd = self.actorLogStd.expand_as(actionMean)
        actionStd = torch.exp(actionLogStd)
        probabilities = Normal(actionMean, actionStd)
        if action is None:
            if evaluation == True:
                action = actionMean
            else:
                action = probabilities.rsample()
        return action, probabilities.log_prob(action).sum(-1), probabilities.entropy().sum(-1), self.critic(x)

    def processObservations(self, x):
        # TODO: Get any observations. Extract 1D and 3D observations out of it and return it
        # if only 1 type of obs, the other one should be returned as tensor of size (0,)
        pass
        # return obs1D, obs3D
