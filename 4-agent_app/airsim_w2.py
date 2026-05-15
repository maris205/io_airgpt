import sys
sys.path.append('..')
import airsim
import math
import numpy as np
import cv2
import base64
import os
from openai import OpenAI
from gdino import GroundingDINOAPIWrapper, visualize
from PIL import Image
import uuid
from smolagents import tool


@tool
def takeoff() -> str:
    """
    Take off the drone. Returns a string indicating whether the action was successful.
    """
    client = airsim.MultirotorClient()#run in some machine of airsim,otherwise,set ip="" of airsim
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()

    return "success"



@tool
def land() -> str:
    """
    Land the drone. Returns a string indicating whether the action was successful.
    """
    client = airsim.MultirotorClient()#run in some machine of airsim,otherwise,set ip="" of airsim
    client.landAsync().join()

    return "success"