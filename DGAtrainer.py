# Dont make agent reset at failure, either it will make them not have failures or learn to recover. We'll learn for the amount of steps anyway

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import *
import copy

#TODO: Make population size independent on environment population. Just iterate through all steps until population has been tested.

# def fitnessBasedSelection(fitnessScores, N, temperature=0.1):
#     minFitness = np.min(fitnessScores)
#     if minFitness < 0:
#         fitnessScores = fitnessScores - minFitness 

#     normFitness = fitnessScores / np.sum(fitnessScores)
#     adjustedFitness = np.exp(normFitness / temperature)
#     adjustedFitness /= np.sum(adjustedFitness)
#     divisions = np.cumsum(adjustedFitness)
#     selected = []
#     for _ in range(N):
#         rand = np.random.rand()
#         index = np.searchsorted(divisions, rand)
#         selected.append(index)
#     return selected

def fitnessBasedSelection(fitness, N, temperature=0.1):
    fitness = torch.softmax(fitness * (1 - temperature), dim=-1)
    print(f"Fitness after softmax {fitness}")
    distribution = Categorical(fitness)
    selected = distribution.sample((N,))
    return selected.tolist()

class Individual(nn.Module):
    def __init__(self, envSpecs, architecture = {"preActor1D": [512, 256], "preActor3D": [16, 32, 32], "actorContinuous": [256], "actorDiscrete": [256]},
                 cActionLowBound = -1, cActionHighBound = 1):
        
        super(Individual, self).__init__()
        self.envSpecs = envSpecs
        self.obsSize1D, self.obsSize3D = getObsSizes(self.envSpecs)
        self.obsChannels3D = self.obsSize3D[0] # first shape dim is channels
        self.continuousActionSize = self.envSpecs["ContinuousActions"]
        self.using1Dobs = self.obsSize1D > 0
        self.using3Dobs = sum(self.obsSize3D) > 0
        self.usingDiscreteActions = len(self.envSpecs["DiscreteActions"]) > 0
        self.usingContinuousActions = self.continuousActionSize > 0
        self.architecture = architecture
        self.preActor3DoutputSize = 0
        assert self.using1Dobs or self.using3Dobs, "No 1D or 3D observations and you expect it to work?!?!?!?"
        assert self.usingDiscreteActions or self.usingContinuousActions, "We have to use either continuous or discrete actions"

        if self.using1Dobs:      
            self.preActor1D = getSequentialModel1D(self.obsSize1D, architecture["preActor1D"][:-1], architecture["preActor1D"][-1], finishWithActivation=True)
            
        if self.using3Dobs:
            self.preActor3D = getSequentialModel3D(self.obsChannels3d, architecture["preActor3D"])
            self.preActor3DoutputSize = calculateConvNetOutputSize(self.preActor3D, self.obsSize3D)

        if self.usingContinuousActions:
            self.actorContinuous = getSequentialModel1D(architecture["preActor1D"][-1] + self.preActor3DoutputSize, architecture["actorContinuous"], self.continuousActionSize)
            self.actorLogStd = nn.Linear(architecture["preActor1D"][-1] + self.preActor3DoutputSize, self.continuousActionSize)
            self.register_buffer("continuousActionScale", torch.tensor((cActionHighBound - cActionLowBound) / 2.0, dtype=torch.float32))
            self.register_buffer("continuousActionBias", torch.tensor((cActionHighBound + cActionLowBound) / 2.0, dtype=torch.float32))

        if self.usingDiscreteActions:
            self.actorDiscrete = getSequentialModel1D(architecture["preActor1D"][-1] + self.preActor3DoutputSize, architecture["actorDiscrete"], sum(self.envSpecs["DiscreteActions"]))
        
    # TODO: not handling action masks yet
    @torch.no_grad()
    def getDiscreteAction(self, x, mask=torch.tensor([1], device=device), processObs=True):
        if processObs:
            obs1D, obs3D = processObservations(x)
            observationFeatures = self.getObservationFeaturesForActor(obs1D, obs3D)
        else:
            observationFeatures = self.getObservationFeaturesForActor(x[0], x[1])

        unsplitLogits = self.actorDiscrete(observationFeatures)
        unsplitLogits = torch.mul(unsplitLogits, mask)
        splitLogits = torch.split(unsplitLogits, list(self.envSpecs["DiscreteActions"]), dim=-1)
        actionDistributions = [Categorical(logits=logits) for logits in splitLogits]
        action = torch.stack([distribution.sample() for distribution in actionDistributions])
        return action.T

    @torch.no_grad()
    def getContinuousAction(self, x, processObs=True):
        if processObs:
            obs1D, obs3D = processObservations(x)
            observationFeatures = self.getObservationFeaturesForActor(obs1D, obs3D)
        else:
            observationFeatures = self.getObservationFeaturesForActor(x[0], x[1])

        # We always evaluate, so no distribution, just mean
        actionMean = self.actorContinuous(observationFeatures)
        actionSampleTanh = torch.tanh(actionMean)
        action = actionSampleTanh * self.continuousActionScale + self.continuousActionBias
        return action

    def getObservationFeaturesForActor(self, obs1D, obs3D):
        if self.using1Dobs and self.using3Dobs:
            return torch.cat((self.preActor1D(obs1D), self.preActor3D(obs3D)), -1)
        elif self.using1Dobs:
            output = self.preActor1D(obs1D)
            return output
        elif self.using3Dobs:
            return self.preActor3D(obs3D)

    def produceOffspring(self):
        offspring = copy.deepcopy(self).to(device)
        mutateNet(offspring)
        return offspring

# def mutateNet(network, mutationRate=0.01, mutationStrength=0.025):
#     for param in network.parameters():
#         if torch.rand(1).item() < mutationRate:
#             mutation = torch.randn_like(param, device=device) * mutationStrength
#             # print(f"Mutating a parameter {param} by {mutation}")
#             param.data.add_(mutation)
    
def mutateNet(network, mutation_rate=0.01, mutation_strength=0.1):
    """
    Mutates a subset of the parameters of a neural network.

    Args:
        network (nn.Module): The neural network to be mutated.
        mutation_rate (float): The fraction of parameters to mutate.
        mutation_strength (float): The standard deviation of the normal distribution from which
                                   the random mutation values are sampled.
    """
    # Flatten all parameters into a single list
    all_params = [p for p in network.parameters()]
    total_params = sum(p.numel() for p in all_params)
    
    # Determine number of parameters to mutate
    num_to_mutate = int(total_params * mutation_rate)
    
    # Generate random indices to mutate
    indices_to_mutate = random.sample(range(total_params), num_to_mutate)
    
    # Mutate the selected parameters
    flat_params = torch.cat([p.view(-1) for p in all_params])
    for idx in indices_to_mutate:
        flat_params[idx].data.add_(torch.randn(1, device=device).item() * mutation_strength)

    # Reassign the mutated parameters back to the original shapes
    start = 0
    for param in all_params:
        numel = param.numel()
        param.data = flat_params[start:start + numel].view(param.shape).data
        start += numel

def initializePopulation(envSpecs, N, architectures = []):
    population = []
    for i, behaviorName in enumerate(envSpecs):
        for _ in range(N//len(envSpecs)):
            if architectures:
                population.append(Individual(envSpecs[behaviorName], architectures[i]))
            else:
                population.append(Individual(envSpecs[behaviorName]))
    return population

seed = 1
totalAgentsCount = 128
nrGenerations = 1000
generationLength = 100

seedEverything(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# env = UnityInterface("Builds\\Windows\\Ball3D\\UnityEnvironment", seed=seed)              # 1D obs only, continuous action of size 2. Rewards: 0.1 for every step, -1 for fail, 100 is the max episodic return
# env = UnityInterface("Builds\\Windows\\Crawler\\UnityEnvironment", seed=seed)             # 1D obs only, continuous action of size 8
# env = UnityInterface("Builds\\Windows\\Crawlerx01\\UnityEnvironment", seed=seed)             # 1D obs only, continuous action of size 8, x01 version, all scaled with 0.1, so different physics
# env = UnityInterface("Builds\\Windows\\Crawlerx01 x128\\UnityEnvironment", seed=seed)             # 1D obs only, continuous action of size 8, x01 version, all scaled with 0.1, so different physics
# env = UnityInterface("Builds\\Windows\\PushBlock\\UnityEnvironment", seed=seed)           # 1D obs only, discrete action of size (7). Rewards: 5 for win, -0.001 for every step
# env = UnityInterface("Builds\\Windows\\PushBlockMov\\UnityEnvironment", seed=seed)           # 1D obs only, discrete action of size (7). Rewards: 5 for win, -0.001 for every step, mod with roughly 0.001 while moving the block
env = UnityInterface("Builds\\Windows\\PushBlock x128\\UnityEnvironment", seed=seed)           # 1D obs only, discrete action of size (7). Rewards: 5 for win, -0.001 for every step, mod with roughly 0.001 while moving the block
# env = UnityInterface("Builds\\Windows\\WallJump\\UnityEnvironment", seed=seed)            # 1D obs only, discrete action of size (3, 3, 3, 2)	# env = UnityInterface("Builds\\Windows\\WallJump\\UnityEnvironment", seed=seed)
# env = UnityInterface("Builds\\Windows\\Worm\\UnityEnvironment", seed=seed)

# env = UnityInterface("Builds/Linux/Ball3D/Ball3D", seed=seed)                             # 1D obs only, continuous action of size 2. Rewards: 0.1 for every step, -1 for fail, 100 is the max episodic return
# env = UnityInterface("Builds/Linux/PushBlock/PushBlock", seed=seed)                       # 1D obs only, discrete action of size (7). Rewards: 5 for win, -0.001 for every step
# env = UnityInterface("Builds/Linux/WallJump/WallJump", seed=seed)                         # 1D obs only, discrete action of size (3, 3, 3, 2)	# env = UnityInterface("Builds/Linux/WallJump/WallJump", seed=seed)

# env = UnityInterface(None, seed=seed)

specs = env.getSpecs()
print(f"{specs}")
behaviorNames = env.getBehaviorNames()

fitness = torch.zeros(totalAgentsCount, dtype=torch.float32)
# terminated = np.zeros(totalAgentsCount, dtype=np.bool_) # Unused for now
population = initializePopulation(specs, totalAgentsCount)

for generation in range(nrGenerations):
    for step in range(generationLength):
        for behavior in behaviorNames:
            decisionSteps, terminalSteps = env.getSteps(behavior)

            # print(f"Generation {generation} step {step} population: {population}")

            if len(decisionSteps) > 0:
                actionsC, actionsD = [], []
                for agentIndex in decisionSteps:
                    fitness[agentIndex] += decisionSteps[agentIndex].reward
                    # print(f"current reward: {decisionSteps[agentIndex].reward}")
                    if population[agentIndex].usingContinuousActions:
                        actionsC.append(population[agentIndex].getContinuousAction([decisionSteps[agentIndex].obs]))
                    if population[agentIndex].usingDiscreteActions:
                        actionsD.append(population[agentIndex].getDiscreteAction([decisionSteps[agentIndex].obs]))

                for agentIndex in terminalSteps:
                    # print(f"current terminal reward: {decisionSteps[agentIndex].reward}")
                    fitness[agentIndex] += terminalSteps[agentIndex].reward

                env.setActions(behavior, torch.cat(actionsC, 0).detach().cpu().numpy() if actionsC else None, torch.cat(actionsD, 0).detach().cpu().numpy() if actionsD else None)
        env.step()
    env.reset()

    chosenParents = fitnessBasedSelection(fitness, len(population)//2, temperature=0.1)

    newChildren = []
    for chosenParent in chosenParents:
        print(f"We chose a parent {chosenParent:>4} with fitness of {fitness[chosenParent]:6.3f}")
        newChildren.append(population[chosenParent].produceOffspring())
    
    # print(f"Generation {generation} has been tested. Producing offspring")

    sortedIndices = torch.argsort(fitness, descending=True)
    population = [population[i] for i in sortedIndices]
    population = population[:len(population)//2] + newChildren
    # print(f"New population with best performers and mutated offspring: {population}")

    print(f"Fitness for generation {generation}: {fitness}")
    fitness.fill_(0.0)

env.close()