import numpy as np
import pandas as pd
from scipy import stats
import yaml

# Read YAML file
with open('./parameters.yaml', 'r') as file_descriptor:
    parameters = yaml.load(file_descriptor)


def rescale_action_space(scale_factor, action):
    action_temp = np.zeros(8)
    action_temp[parameters['model']['ppo_acting_joints']] = action * scale_factor

    if action_temp[7] > 0:
        action_temp[7] = 25
    else:
        action_temp[7] = -25
    return action_temp


class Occlusion:
    def __init__(self):
        self.duration = np.random.randint(50, 300)
        self.timestep = -1
        self.occlusion = False

    def update_duration(self):
        self.timestep += 1

        if self.timestep == self.duration:
            self.duration = np.random.randint(1, 500)
            self.occlusion = False
            self.timestep = -1
        else:
            self.occlusion = True
