import numpy as np
import torch
from mlagents_envs.environment import UnityEnvironment

class UnityInterface():
    def __init__(self, envName):
        self.env = UnityEnvironment(file_name=envName, side_channels=[])
        self.env.reset()
        self.agentsNames = list(self.env.behavior_specs)
        self.specs = {}
        for agent in self.agentsNames:
            self.specs[agent] = self.env.behavior_specs[agent]
        self.decisionSteps, self.terminalSteps = self.env.get_steps()
        
