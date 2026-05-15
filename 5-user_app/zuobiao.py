# -*- coding: utf-8 -*-
# @Time    : 2024/9/4  11:31
# @Author  : cxq213@foxmail.com
# @File    : zuobiao.py
# @Software: PyCharm
# @Describe: Coordinate utilities
# -*- encoding:utf-8 -*-
import sys
sys.path.append('..')
import airsim
import math
import numpy as np

"""
Function: Get coordinates by object name in the scene
"""

# Create AirSim client
import time
import cv2
import matplotlib.pyplot as plt

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()


# Get names of all objects in the scene
all_objects = client.simListSceneObjects()

# Assume the ball's name contains "Ball" (typically the name may include "Ball")
# object_name = "Ball"
object_name = "HouseForAirsim_C_1"
# object_name = "SK_West_Tank_M1A1Abrams2_25"
for obj_name in all_objects:
    if object_name in obj_name:
        pose = client.simGetObjectPose(obj_name)
        print(f"Object Name: {obj_name}")
        print(f"Position: x = {pose.position.x_val}, y = {pose.position.y_val}, z = {pose.position.z_val}")
        print(f"Position: {pose.position.x_val},{pose.position.y_val}, {pose.position.z_val}")
        print(f"Orientation: pitch = {pose.orientation.x_val}, roll = {pose.orientation.y_val}, yaw = {pose.orientation.z_val}")
