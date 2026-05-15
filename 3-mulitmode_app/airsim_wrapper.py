import sys
sys.path.append('..')
import airsim
import math
import numpy as np
import cv2
import base64
import json
import re
from openai import OpenAI

api_key="bc859308-e437-4129-9102-ce285637a63c" # Use your own key

objects_dict = {
"cola": "StaticMesh",
"orchid": "orchid05",
"coconut water": "a_polySurface17",
"rubber duck": "Yellowduck",
"mirror": "Mirror_06",
"square table": "desk",
}


class AirSimWrapper:
def __init__(self):
# Drone client
self.client = airsim.MultirotorClient()#run in some machine of airsim,otherwise,set ip="" of airsim
self.client.confirmConnection()
self.client.enableApiControl(True)
self.client.armDisarm(True)


# LLM client
self.llm_client = OpenAI(
api_key=api_key,
base_url="https://api.intelligence.io.solutions/api/v1",
)


def takeoff(self):
"""
takeoff the drone
"""
self.client.takeoffAsync().join()

def land(self):
"""
land the drone
"""
self.client.landAsync().join()


def get_drone_position(self):
"""
get the current position of the drone
:return: position, the current position of the drone
"""
pose = self.client.simGetVehiclePose()
yaw_degree = self.get_yaw() # angle in degree
return [pose.position.x_val, pose.position.y_val, pose.position.z_val,yaw_degree]

def fly_to(self, point):
"""
fly the drone to a specific point
:param point: the target point
"""
if point[2] > 0:
self.client.moveToPositionAsync(point[0], point[1], -point[2], 1).join()
else:
self.client.moveToPositionAsync(point[0], point[1], point[2], 1).join()



def fly_path(self, points):
"""
fly the drone along a specific path
:param points: the path
"""
airsim_points = []
for point in points:
if point[2] > 0:
airsim_points.append(airsim.Vector3r(point[0], point[1], -point[2]))
else:
airsim_points.append(airsim.Vector3r(point[0], point[1], point[2]))
#self.client.moveOnPathAsync(airsim_points, 5, 120, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0), 20, 1).join()
self.client.moveOnPathAsync(airsim_points, 1).join()


def set_yaw(self, yaw_degree):
"""
set the yaw angle of the drone
"""
self.client.rotateToYawAsync(yaw_degree, 5).join()

def get_yaw(self):
"""
get the yaw angle of the drone
:return: yaw_degree, the yaw angle of the drone in degree
"""
orientation_quat = self.client.simGetVehiclePose().orientation
yaw = airsim.to_eularian_angles(orientation_quat)[2] # get the yaw angle
yaw_degree = math.degrees(yaw)
return yaw_degree # return the yaw angle in degree

def get_position(self, object_name):
"""
get the position of a specific object
:param object_name: the name of the object
:return: position, the position of the object
"""
query_string = objects_dict[object_name] + ".*"
object_names_ue = self.client.simListSceneObjects(query_string)
if len(object_names_ue) == 0:
raise ValueError(f"Object not found in scene: {object_name} (query: {query_string})")
pose = self.client.simGetObjectPose(object_names_ue[0])

#yaw_degree = math.degrees(pose.orientation.z_val) #angle in degree

orientation_quat = pose.orientation
yaw = airsim.to_eularian_angles(orientation_quat)[2] # get the yaw angle
yaw_degree = math.degrees(yaw)


return [pose.position.x_val, pose.position.y_val, pose.position.z_val, yaw_degree]

def look_at(self, yaw_degree):
self.set_yaw(yaw_degree)

def turn_left(self):
"""
Turn left by 10 degrees
:return:
"""
yaw_degree = self.get_yaw()
yaw_degree = yaw_degree - 10
self.set_yaw(yaw_degree)


def turn_right(self):
"""
Turn right by 10 degrees
:return:
"""
yaw_degree = self.get_yaw()
yaw_degree = yaw_degree + 10
self.set_yaw(yaw_degree)

def forward(self):
"""
Move forward by 1 meter (too little may not register movement)
:return:
"""
step_length = 1
cur_position = self.get_drone_position()
yaw_degree = cur_position[3]
# Convert angle to radians
yaw = math.radians(yaw_degree)
# Move forward by step_length
x = cur_position[0] + step_length*math.cos(yaw)
y = cur_position[1] + step_length*math.sin(yaw)
z = cur_position[2]
self.fly_to([x, y, z])


def reset(self):
self.client.reset()

def cv2_to_base64(self, image, format='.png'):
"""Convert an OpenCV numpy array image to a Base64 string"""
# Encode to byte stream
success, buffer = cv2.imencode(format, image)
if not success:
raise ValueError("Image encoding failed, please check the format parameter")

# Convert to Base64
img_bytes = buffer.tobytes()
return base64.b64encode(img_bytes).decode('utf-8')

def get_image(self):
"""
Get the front camera rendered image
:return:
"""
camera_name = '0' # Front center: 0, Bottom center: 3
image_type = airsim.ImageType.Scene # Color image: airsim.ImageType.Scene, Infrared
response = self.client.simGetImage(camera_name, image_type, vehicle_name='') # simGetImage API usage:


img_bgr = cv2.imdecode(np.array(bytearray(response), dtype='uint8'), cv2.IMREAD_UNCHANGED) # Read from binary image data
img = cv2.cvtColor(img_bgr, cv2.COLOR_RGBA2RGB) # 43

#print("image shape:", img.shape)
return img

def look(self):
"""
Get the front camera rendered image and return a list of main objects in the image
:return: string, object names separated by commas
"""
# Step 1: Read camera image (already in RGB)
rgb_image = self.get_image()

# Convert to base64 PNG image
base64_str = self.cv2_to_base64(rgb_image, ".png") # png or '.jpg'

# Step 2: Perform image understanding
# Image input:
response = self.llm_client.chat.completions.create(
model="moonshotai/Kimi-K2.6",
messages=[
{
"role": "user",
"content": [
{"type": "text", "text": "What objects are in this image? Please list only the names of common, clearly visible objects, separated by commas."},
{
"type": "image_url",
"image_url": {
# Use Base64 encoded local image; make sure img/png or img/jpg is correct
"url": f"data:image/png;base64,{base64_str}"
}
},
],
}
],
temperature=0.01
)

content = response.choices[0].message.content
return content

def detect(self, object_names):
"""
Perform object detection on the current camera image, returning detected categories and bounding boxes
:param object_names: Detection targets (comma-separated string, e.g., 'duck, cola')
:return: obj_id_list, obj_locs, vis_img(None)
"""
rgb_image = self.get_image()
base64_str = self.cv2_to_base64(rgb_image, ".png")

prompt = (
f"Please detect the following objects in the image: {object_names}\n"
"For each detected object, return a JSON array in this format:\n"
'[{"category": "object name", "bbox": [x1, y1, x2, y2]}]\n'
"where bbox is the bounding box pixel coordinates (top-left x1,y1, bottom-right x2,y2). Return only the JSON array, no other text."
)

response = self.llm_client.chat.completions.create(
model="moonshotai/Kimi-K2.6",
messages=[
{
"role": "user",
"content": [
{"type": "text", "text": prompt},
{
"type": "image_url",
"image_url": {"url": f"data:image/png;base64,{base64_str}"}
},
],
}
],
temperature=0.01
)

content = response.choices[0].message.content
match = re.search(r'\[.*\]', content, re.DOTALL)
objects = json.loads(match.group()) if match else []

obj_id_list = [obj['category'] for obj in objects]
obj_locs = [obj['bbox'] for obj in objects]

vis_image = rgb_image.copy()
for obj in objects:
x1, y1, x2, y2 = [int(v) for v in obj['bbox']]
cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(vis_image, obj['category'], (x1, max(y1 - 8, 0)),
cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
from PIL import Image as _PILImage
img_with_box = _PILImage.fromarray(vis_image)

# #region agent log
import time as _t, os as _os, json as _j
_log_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'debug-306aab.log')
with open(_log_path, 'a', encoding='utf-8') as _f:
_f.write(_j.dumps({"sessionId":"306aab","timestamp":int(_t.time()*1000),"location":"airsim_wrapper.py:detect","message":"detect post-fix","data":{"obj_id_list":obj_id_list,"obj_locs_count":len(obj_locs),"img_with_box_type":str(type(img_with_box))},"hypothesisId":"A-B","runId":"post-fix"}) + '\n')
print(f"[DEBUG] detect() new code running, found: {obj_id_list}")
# #endregion

return obj_id_list, obj_locs, img_with_box


def get_distance(self):
"""
get the distance between the quadcopter and the nearest obstacle
:return: distance, the distance between the quadcopter and the nearest obstacle
"""
distance = 100000000

pose = self.client.simGetVehiclePose() # get the current pose of the quadcopter
v_p = [pose.position.x_val, pose.position.y_val, pose.position.z_val]

# get lidar data
lidarData = self.client.getLidarData()
if len(lidarData.point_cloud) < 3:
return distance # if no points are received from the lidar, return a big distance as 100000000

points = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
points = np.reshape(points, (int(points.shape[0] / 3), 3))
distance_list = []
for p in points:
distance = np.linalg.norm(np.array(v_p) - p)
distance_list.append(distance)

distance = min(distance_list)
return distance

if __name__ == "__main__":
aw = AirSimWrapper()
aw.takeoff()
object = aw.look()
print(object)
print("done")
