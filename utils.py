import numpy as np
import torch
from mlagents_envs.environment import UnityEnvironment, ActionTuple

class UnityInterface():
    def __init__(self, envName=None):
        self.env = UnityEnvironment(file_name=envName, side_channels=[])
        self.env.reset()
        self.behaviorNames = list(self.env.behavior_specs)
        self.specs = {}
        for behavior in self.behaviorNames:
            behaviorSpecs = {}
            obsSpecs = self.env.behavior_specs[behavior].observation_specs
            behaviorSpecs["AgentsCount"] = self.countAgents(behavior)
            behaviorSpecs["Observations"] = [obsSpecs[i].shape for i in range(len(obsSpecs))]
            behaviorSpecs["ContinuousActionSize"] = self.env.behavior_specs[behavior].action_spec.continuous_size
            behaviorSpecs["DiscreteActionSize"] = self.env.behavior_specs[behavior].action_spec.discrete_size # Not needed
            behaviorSpecs["DiscreteBranches"] = self.env.behavior_specs[behavior].action_spec.discrete_branches
            self.specs[behavior] = behaviorSpecs
        
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

    def step(self, actionsDict):
        for agent, action in actionsDict.items():
            action = ActionTuple(continuous=action["continuous"], discrete=action["discrete"])
            self.env.set_actions(agent, action)
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
