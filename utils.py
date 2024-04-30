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

    # def getInitialObservations(self, bufferList):
    #     for behavior in self.behaviorNames:
    #         decisionSteps, _ = self.env.get_steps(behavior)
    #         for agentNr in decisionSteps:
    #             bufferList[agentNr] = decisionSteps[agentNr].obs
    #     return bufferList

def layerInit(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

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
        # TODO Getting a warning with this approach.. Could rework
        # Only batched pass
        allObs1D, allObs3D = [], []
        for observation in x:
            obs1D = torch.zeros((0,))
            obs3D = torch.zeros((0,))
            for observationElement in observation:
                observationElement = torch.from_numpy(observationElement)
                # print(f"Observation of shape: {list(observationElement.shape)}")
                if len(list(observationElement.shape)) == 1:
                    # print(f"BATCHED So, dimensionality is 1 and we add it to obs1D of shape: {list(obs1D.shape)}")
                    obs1D = torch.cat((obs1D, observationElement))
                    # print(f"Making it of shape: {list(obs1D.shape)}")
                elif len(list(observationElement.shape)) == 3:
                    # print(f"BATCHED So, dimensionality is 3 and we add it to obs1D of shape: {list(obs3D.shape)}")
                    obs3D = torch.cat((obs3D, observationElement))
                    # print(f"Making it of shape: {list(obs3D.shape)}")
                else:
                    print(f"Unexpected {len(list(observationElement.shape))}-dimensional observation")
            allObs1D.append(obs1D.clone())
            allObs3D.append(obs3D.clone())
        # TODO: Put it on device?
        # print(f"processObservations returning obs1D of shape {list(obs1D.shape)} abd obs3D of shape {list(obs3D.shape)}")
        # print(f"Will be stacking lists allObs1D and allObs3D: {allObs1D}, {allObs3D}")
        # print(f"Outputting stacked allObs1D and allObs3D of shapes: {torch.stack(allObs1D).shape}, {torch.stack(allObs3D).shape}")
        return torch.stack(allObs1D).to(device), torch.stack(allObs3D).to(device)

class QNetwork(nn.Module):
    def __init__(self, envSpecs):
        super(QNetwork, self).__init__()
        self.envSpecs = envSpecs
        self.obsSize1D, self.obsSize3D = getObsSizes(self.envSpecs)
        self.obsChannels3D = self.obsSize3D[0] # first shape dim is channels
        self.continuousActionSize = self.envSpecs["ContinuousActions"]
        self.preCritic1DoutputSize = 0
        self.preCritic3DoutputSize = 0
        self.using1Dobs = self.obsSize1D > 0
        self.using3Dobs = sum(self.obsSize3D) > 0
        self.usingDiscreteActions = len(self.envSpecs["DiscreteActions"]) > 0
        self.usingContinuousActions = self.continuousActionSize > 0

        if self.using1Dobs:
            self.preCritic1DoutputSize = 64
            self.preCritic1D = nn.Sequential(
                layerInit(nn.Linear(self.obsSize1D, 128)), nn.Tanh(),
                layerInit(nn.Linear(128, self.preCritic1DoutputSize)), nn.Tanh())
        
        if self.using3Dobs:
            self.preCritic3D = nn.Sequential(
                layerInit(nn.Conv2d(self.obsChannels3D, 16, 7, stride=4)), nn.Tanh(),
                layerInit(nn.Conv2d(16, 32, 5, stride=2)), nn.Tanh(),
                layerInit(nn.Conv2d(32, 32, 3, stride=1)), nn.Tanh(), nn.Flatten())
            self.preCritic3DoutputSize = calculateConvNetOutputSize(self.preCritic3D, self.obsSize3D)

        self.criticFinal = nn.Sequential(layerInit(nn.Linear(self.preCritic1DoutputSize + self.preCritic3DoutputSize + self.getTotalActionSize(), 1), std=0.01), nn.Flatten())
    
    def forward(self, x, actionsContinuous=None, actionsDiscrete=None):
        obs1D, obs3D = processObservations(x)
        standardObsFeatures = self.getObservationFeaturesForCritic(obs1D, obs3D)
        actionObsFeatures = self.getActionRepresentationForInput(actionsContinuous, actionsDiscrete)
        # print(f"Will try to cat standardObsFeatures of shape {standardObsFeatures.shape} and actionObsFeatures of shape: {actionObsFeatures.shape}")
        return self.criticFinal(torch.cat((standardObsFeatures, actionObsFeatures), -1))

    def getObservationFeaturesForCritic(self, obs1D, obs3D):
        netsOutputs = []
        if self.using1Dobs:
            netsOutputs.append(self.preCritic1D(obs1D))
        if self.using3Dobs:
            netsOutputs.append(self.preCritic3D(obs3D))
        return torch.cat(netsOutputs, -1)

    def getTotalActionSize(self):
        return self.continuousActionSize + sum(self.envSpecs["DiscreteActions"])
    
    def getActionRepresentationForInput(self, actionsContinuous=None, actionsDiscrete=None):
        featuresList = []
        # print(f"func getActionRepresentationForInput, where C:{actionsContinuous}, D:{actionsDiscrete}")
        if actionsContinuous != None:
            # print(f"actionsContinuous: {actionsContinuous}")
            # print(f"actionsContinuous.shape: {actionsContinuous.shape}")
            featuresC = torch.tensor(actionsContinuous)
            featuresList.append(featuresC)

        if actionsDiscrete != None:
            # print(f"actionsDiscrete: {actionsDiscrete}")
            onehots = []
            for i, discreteSize in enumerate(self.envSpecs["DiscreteActions"]):
                # print(f"i: {i}, analyzed discrete action size: {discreteSize}")
                # print(f"onehotting actionsDiscrete {actionsDiscrete}")
                onehots.append(F.one_hot(actionsDiscrete[:, i], discreteSize))
            featuresList.append(torch.cat(onehots, -1))
    
        features = torch.cat(featuresList, -1)
        # print(f"Final input respresentation (first 3):\n{features[:3]}")
        return features


class SoftQNetwork(nn.Module):
    def __init__(self, envSpecs):
        super(SoftQNetwork, self).__init__()
        self.QNet1 = QNetwork(envSpecs).to(device)
        self.QNet2 = QNetwork(envSpecs).to(device)
        self.QNet1Target = QNetwork(envSpecs).to(device)
        self.QNet2Target = QNetwork(envSpecs).to(device)
        self.QNet1Target.load_state_dict(self.QNet1.state_dict())
        self.QNet2Target.load_state_dict(self.QNet2.state_dict())

        self.QNetsOptimizer = optim.Adam(list(self.QNet1.parameters()) + list(self.QNet2.parameters()), lr=1e-3)  
        self.tau = 0.005

LOG_STD_MAX = 2
LOG_STD_MIN = -5

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
            self.preActor1DoutputSize = 256
            self.preActor1D = nn.Sequential(
                layerInit(nn.Linear(self.obsSize1D, 256)), nn.Tanh(),
                layerInit(nn.Linear(256, self.preActor1DoutputSize)), nn.Tanh())
            
        if self.using3Dobs:
            self.preActor3D = nn.Sequential(
                layerInit(nn.Conv2d(self.obsChannels3D, 16, 7, stride=4)), nn.Tanh(),
                layerInit(nn.Conv2d(16, 32, 5, stride=2)), nn.Tanh(),
                layerInit(nn.Conv2d(32, 32, 3, stride=1)), nn.Tanh(), nn.Flatten())
            self.preActor3DoutputSize = calculateConvNetOutputSize(self.preActor3D, self.obsSize3D)


        self.usingDiscreteActions = len(self.envSpecs["DiscreteActions"]) > 0
        self.usingContinuousActions = self.continuousActionSize > 0
        assert self.usingDiscreteActions or self.usingContinuousActions, "We have to use either continuous or discrete actions"

        if self.usingContinuousActions:
            self.actorContinuous = nn.Sequential(
                layerInit(nn.Linear(self.preActor1DoutputSize + self.preActor3DoutputSize, 128)), nn.Tanh(),
                layerInit(nn.Linear(128, self.continuousActionSize), std=0.01))        
            self.actorLogStd = nn.Linear(self.preActor1DoutputSize + self.preActor3DoutputSize, self.continuousActionSize)
            self.register_buffer("continuousActionScale", torch.tensor((continuousActionHighBound - continuousActionLowBound) / 2.0, dtype=torch.float32))
            self.register_buffer("continuousActionBias", torch.tensor((continuousActionHighBound + continuousActionLowBound) / 2.0, dtype=torch.float32))


        if self.usingDiscreteActions:
            self.actorDiscrete = nn.Sequential(
                layerInit(nn.Linear(self.preActor1DoutputSize + self.preActor3DoutputSize, 256)), nn.Tanh(),
                layerInit(nn.Linear(256, sum(self.envSpecs["DiscreteActions"])), std=0.01))
            # print(f"So because we have actions defined as {self.envSpecs['DiscreteActions']}, are output discrete layer is of size {sum(self.envSpecs['DiscreteActions'])}")
        
        self.actorOptimizer = optim.Adam(list(self.parameters()), lr=3e-4)

    # TODO: I'd like to break it down so it doesnt calculate logprobs when I need only actions
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
        # probs = torch.stack([torch.exp(categorical.log_prob(a)) for a, categorical in zip(action, multi_categoricals)])
        # entropies = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        # print(f"Discrete logprobs are {logprobs} oftorch.min(QFunction1NextTarget, QFunction2NextTarget) shape {logprobs.shape} and we will sum them along 0: {logprobs.sum(0)} of shape {logprobs.sum(0).shape}")
        return action.T, logprobs.sum(0)
    
    def getContinuousActionAndValue(self, x, evaluation=False):
        obs1D, obs3D = processObservations(x)
        observationFeatures = self.getObservationFeaturesForActor(obs1D, obs3D)
        # print(f"Trying to pass to actorContinuous {self.actorContinuous} features of shape {observationFeatures.shape}")

        actionMean = self.actorContinuous(observationFeatures)
        actionLogStd = torch.tanh(self.actorLogStd(observationFeatures))
        actionLogStd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (actionLogStd + 1)  # From SpinUp / Denis Yarats
        actionStd = actionLogStd.exp()
        distribution = Normal(actionMean, actionStd)

        if evaluation == True:
            actionSample = actionMean
        else:
            actionSample = distribution.rsample()
        
        actionSampleTanh = torch.tanh(actionSample)
        action = actionSampleTanh * self.continuousActionScale + self.continuousActionBias
        logProbs = distribution.log_prob(actionSample)
        logProbs -= torch.log(self.continuousActionScale * (1 - actionSampleTanh.pow(2)) + 1e-6)
        logProbs = logProbs.sum(-1)
        
        return action, logProbs

    def getObservationFeaturesForActor(self, obs1D, obs3D):
        # print(f"Getting obsFeatues for actor with obs1D of shape {obs1D.shape} and obs3D of shape {obs3D.shape}")
        netsOutputs = []
        if self.using1Dobs:
            # print(f"Trying to feed preActor1D an input of shape {list(obs1D.shape)} while obsSize1D is {self.obsSize1D}")
            netsOutputs.append(self.preActor1D(obs1D))
            # print(f"Appending to outputs preActor1D outputs of shape {self.preActor1D(obs1D).shape}")
        if self.using3Dobs:
            netsOutputs.append(self.preActor3D(obs3D))
            # print(f"Appending to outputs preActor3D outputs of shape {self.preActor3D(obs3D).shape}")
        return torch.cat(netsOutputs, -1)
    
class Memory(object):
    def __init__(self, capacity, fieldNames=["observations", "actionsContinuous", "actionsDiscrete", "rewards", "dones", "nextObservations"]):
        self.fieldNames = namedtuple("fieldNames", fieldNames)
        self.memory = deque(maxlen=capacity)

    def push(self, observation, actionContinuous, actionDiscrete, reward, done, nextObservation):
        # print(f"Appending a memory: { self.fieldNames(observation, actionContinuous, actionDiscrete, reward, done, nextObservation)}")
        self.memory.append(self.fieldNames(observation, actionContinuous, actionDiscrete, reward, done, nextObservation))

    def sample(self, batchSize):
        sampledEntries = random.sample(self.memory, batchSize)
        return self.fieldNames(*zip(*sampledEntries))

    def __len__(self):
        return len(self.memory)
    