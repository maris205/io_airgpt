# -*- coding: utf-8 -*-
# @Time    : 2023/12/23  22:24
# @Author  : mariswang@rflysim
# @File    : tello_wrapper.py
# @Software: PyCharm
# @Describe: 
# -*- encoding:utf-8 -*-


import sys
import threading
import time


from djitellopy import Tello
import math
import numpy as np
import io
import cv2
import random
from PIL import Image


BOX_TRESHOLD = 0.25
TEXT_TRESHOLD = 0.25



class TelloWrapper:
    def __init__(self):
        self.client = Tello()
        self.client.connect()


        #启动视频流
        self.client.streamon()  # 开启视频传输
        t = threading.Thread(target=self.get_stream)
        t.setDaemon(True)
        t.start()

        self.head_img = None #tello 前置摄像头



    def get_stream(self):
        while True:
            self.head_img = self.client.get_frame_read().frame
            time.sleep(0.01)

    def takeoff(self):
        self.client.takeoff()

    def land(self):
        self.client.land()


    def turn_left(self,degree=10):
        """
        左转degree度
        :return:
        """
        degree = 10*degree #tello base is 0.1度
        self.client.rotate_counter_clockwise(degree)


    def turn_right(self,degree=10):
        """
        右转degree度
        :return:
        """
        degree = 10*degree #tello base is 0.1度
        self.client.rotate_clockwise(degree)


    def forward(self, distance):
        """
        向前移动, 太少了不动
        distance: 距离，米
        :return:
        """
        distance = distance*100 # tello base is 1cm
        self.client.move_forward(distance) #向前移动50cm


    def back(self, distance):
        """
        向后移动, 太少了不动
        distance: 距离，米
        :return:
        """
        distance = distance*100 # tello base is 1cm
        self.client.move_back(distance) #向前移动50cm


    def up(self, distance):
        """
        向上移动, 太少了不动
        distance: 距离，米
        :return:
        """
        distance = distance*100 # tello base is 1cm
        self.client.move_up(distance) #向前移动50cm


    def down(self, distance):
        """
        向下移动, 太少了不动
        distance: 距离，米
        :return:
        """
        distance = distance*100 # tello base is 1cm
        self.client.move_down(distance) #向前移动50cm
        
        
    def get_image(self):
        """
        获得前置摄像头渲染图像
        :return:
        """
        return self.head_img

    def get_drone_state(self):
        """
        获得无人机状态,
        :return:{'pitch': int, 'roll': int, 'yaw': int}
        """
        return self.client.query_attitude()



        
    

    
    def ob_objects(self,obj_name_list):
        """
        注意需要先执行get_image，
        在图像 img 上运行对象检测模型，获得目标列表 [ <对象名称、距离、角度（以度为单位）>,...]
        :return:对象名称列表、对象信息列表、bbox图
        """
        pass


    def ob_objects_llm(self,obj_name_list):
        """
        注意需要先执行get_image，为llm提供观测结果
        在图像 img 上运行对象检测模型，获得目标列表 [ <对象名称、距离、角度（以度为单位）>,...] , 给到llm用于推理
        :return:[ <对象名称、距离、角度（以度为单位）>,...] 如[(门，0.53，22)，(椅子，4.84，-21)]
        """
        #获得识别结果
        ob_list, final_obj_list, annotated_frame = self.ob_objects(obj_name_list)

        final_result = []

        for obj_info in final_obj_list:
            item = (obj_info[0], obj_info[1], obj_info[3]) #obj_name, camera_distance, angel_degree
            final_result.append(item)

        return final_result








