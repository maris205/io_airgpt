import sys
import sys
sys.path.append('..')
import airsim
import math
import numpy as np

"""
can: by scene in, get
"""

# create AirSim 
import time
import cv2
import matplotlib.pyplot as plt

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()

# getscene in has object 
all_objects = client.simListSceneObjects()

print("scene in has object:", all_objects)