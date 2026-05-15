import sys
sys.path.append('..')
import airsim
import math
import numpy as np
import cv2
import base64
import os
from openai import OpenAI
# from gdino import GroundingDINOAPIWrapper, visualize
from PIL import Image
import uuid 
from smolagents import tool
from typing import List,Tuple,Any
from dds_cloudapi_sdk.tasks.v2_task import create_task_with_local_image_auto_resize
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.visualization_util import visualize_result



api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxx" # Use your own API key from io.solutions
gdino_token = "xxxxxxxxxxxxxxxxxxxxxxxxxx" # Use your own token for DINO

objects_dict = {
"cola": "airsim_coca",
"orchid": "airsim_lanhua",
"coconut water": "airsim_yezishui",
"rubber duck": "airsim_duck",
"mirror": "airsim_mirror_06",
"square table": "airsim_fangzhuo",
}


#AirSimWrapper
client = airsim.MultirotorClient()#run in some machine of airsim,otherwise,set ip="" of airsim

# LLM client
llm_client = OpenAI(
api_key=api_key,
base_url="https://api.intelligence.io.solutions/api/v1",
)


@tool
def takeoff() -> str:
"""
Take off the drone. Returns a string indicating whether the action was successful.

Returns:
str: Success status description
"""
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

return "success"

@tool
def land() -> str:
"""
Land the drone. Returns a string indicating whether the action was successful.

Returns:
str: Success status description
"""
client.landAsync().join()

return "success"

@tool
def get_drone_position()->Tuple[float, float, float, float]:
"""
Get the drone's current position and yaw angle.
Return:
Tuple[x, y, z, yaw_degree]: Tuple containing 3D coordinates (x/y/z) and yaw angle (in degrees)
"""
pose = client.simGetVehiclePose()
yaw_degree = get_yaw() # angle in degree
return [pose.position.x_val, pose.position.y_val, pose.position.z_val,yaw_degree]

@tool
def fly_to(point: Tuple[float,float,float,float]) -> str:
"""
fly the drone to a specific point

Args:
point:Tuple[x, y, z, yaw_degree]: Target point, tuple containing 3D coordinates (x/y/z) and yaw angle (in degrees)
"""
if point[2] > 0:
client.moveToPositionAsync(point[0], point[1], -point[2], 1).join()
else:
client.moveToPositionAsync(point[0], point[1], point[2], 1).join()

return "success"


@tool
def fly_path(points: List[Tuple[float, float, float,float]]) -> str:
"""
fly the drone along a specific path

Args:
points: List of path points, each point is a tuple of 3D coordinates (x, y, z) and yaw angle (in degrees)

Returns:
str: Success status description
"""
airsim_points = []
for point in points:
if point[2] > 0:
airsim_points.append(airsim.Vector3r(point[0], point[1], -point[2]))
else:
airsim_points.append(airsim.Vector3r(point[0], point[1], point[2]))
client.moveOnPathAsync(airsim_points, 1).join()
return "success"

@tool
def set_yaw(yaw_degree: float) -> str:
"""
Set the drone's yaw angle

Args:
yaw_degree: Drone yaw angle (in degrees)

Returns:
str: Success status description
"""
client.rotateToYawAsync(yaw_degree, 5).join()

return "success"

@tool
def get_yaw()->float:
"""
get the yaw angle of the drone

Returns:
float: yaw_degree, the yaw angle of the drone in degree
"""
orientation_quat = client.simGetVehiclePose().orientation
yaw = airsim.to_eularian_angles(orientation_quat)[2] # get the yaw angle
yaw_degree = math.degrees(yaw)
return yaw_degree # return the yaw angle in degree

@tool
def get_position(object_name: str)-> Tuple[float,float,float,float]:
"""
get the position of a specific object

Args:
object_name: the name of the object

Returns:
Tuple[float,float,float,float]: position of the object, tuple of 3D coordinates (x, y, z) and yaw angle (in degrees)
"""
query_string = objects_dict[object_name] + ".*"
object_names_ue = []
while len(object_names_ue) == 0:
object_names_ue = client.simListSceneObjects(query_string)
pose = client.simGetObjectPose(object_names_ue[0])

#yaw_degree = math.degrees(pose.orientation.z_val) #angle in degree

orientation_quat = pose.orientation
yaw = airsim.to_eularian_angles(orientation_quat)[2] # get the yaw angle
yaw_degree = math.degrees(yaw)


return [pose.position.x_val, pose.position.y_val, pose.position.z_val, yaw_degree]

@tool
def look_at(yaw_degree: float)->str:
"""
setdrone heading

Args:
yaw_degree: yaw angle (angle) 

Returns:
str: successstatedescription
"""
set_yaw(yaw_degree)
return "success"


@tool
def turn_left()->str:
"""
, 10

Returns:
str: successstatedescription
"""
yaw_degree = get_yaw()
yaw_degree = yaw_degree - 10
set_yaw(yaw_degree)
return "success"

@tool
def turn_right()->str:
"""
, 10

Returns:
str: successstatedescription
"""
yaw_degree = get_yaw()
yaw_degree = yaw_degree + 10
set_yaw(yaw_degree)
return "success"

@tool
def forward()->str:
"""
toward before 1, few not 

Returns:
str: successstatedescription
"""
step_length = 1
cur_position = get_drone_position()
yaw_degree = cur_position[3]
# Convert angle to radians
yaw = math.radians(yaw_degree)
# Move forward 0.1 meters
x = cur_position[0] + step_length*math.cos(yaw)
y = cur_position[1] + step_length*math.sin(yaw)
z = cur_position[2]
fly_to([x, y, z])
return "success"


def reset():
client.reset()


def cv2_to_base64(image, format='.png'):
""" OpenCV within in numpy array Base64 string"""
# Encode as byte stream
success, buffer = cv2.imencode(format, image)
if not success:
raise ValueError("imageencodingfailure, checkformatparameter")

# Convert to Base64
img_bytes = buffer.tobytes()
return base64.b64encode(img_bytes).decode('utf-8')


def get_image():
"""
before cameraimage
:return:
"""
camera_name = '0' # before toward in between 	0, in between 3
image_type = airsim.ImageType.Scene # Color image: airsim.ImageType.Scene, Infrared
response = client.simGetImage(camera_name, image_type, vehicle_name='') # simGetImage API usage:


img_bgr = cv2.imdecode(np.array(bytearray(response), dtype='uint8'), cv2.IMREAD_UNCHANGED) # Read from binary image data
img = cv2.cvtColor(img_bgr, cv2.COLOR_RGBA2RGB) # 43

#print("image shape:", img.shape)
return img

@tool
def look()->str:
"""
before cameraimage, to image in need list

Return:
str: target use 
"""
# Step 1: Read camera image (already RGB)
rgb_image = get_image()

# Convert to Base64 PNG image
base64_str = cv2_to_base64(rgb_image, ".png") # png or '.jpg'

# Step 2: Image understanding
# Image input:
response = llm_client.chat.completions.create(
model="moonshotai/Kimi-K2.6",
messages=[
{
"role": "user",
"content": [
{"type": "text", "text": "image in has target, to i.e., to, target i.e., many target between use "},
{
"type": "image_url",
"image_url": {
# "url": "https://yt-shanghai.tos-cn-shanghai.volces.com/tello.jpg"
# Use Base64-encoded local image, note img/png or img/jpg format
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

@tool
def detect(object_names: str) -> Tuple[List[str], List[Any]]:
"""
for imageobject detection, return class and 

Args:
object_names (str): need to targetclass (, 'duck, cola') 

Returns:
Tuple[List[str], List[Any]]: 
- obj_id_list: objectIDlist
- obj_locs: objectpositioninformation (edge) 
"""
config = Config(gdino_token)
client = Client(config)
rgb_image = get_image()
file_name = f"random_{uuid.uuid4().hex}.png"
cv2.imwrite(file_name, rgb_image)
try:
task = create_task_with_local_image_auto_resize(
api_path="/v2/task/dinox/detection",
api_body_without_image={
"model": "DINO-X-1.0",
"prompt": {"type": "text", "text": object_names},
"targets": ["bbox"],
"bbox_threshold": 0.25,
"iou_threshold": 0.8
},
image_path=file_name
)
client.run_task(task)
result = task.result
obj_id_list = [obj['category'] for obj in result['objects']]
obj_locs = [obj['bbox'] for obj in result['objects']]
return obj_id_list, obj_locs
finally:
if os.path.exists(file_name):
os.remove(file_name)

def detect_with_img(object_names: str)-> Tuple[List[str], List[str], Image.Image]:
"""
in image above runobject detection, returnresult and image

Args:
object_name: need to target, note this functioninput targetobject_name can is, ifneed tosearch is in, need to below 

Returns:
Tuple[List[str], List[List[float]]]:
- objectlist
- each object edgelist (format: [xmin, ymin, xmax, ymax]) 
- PILimageobject
"""
config = Config(gdino_token)
client = Client(config)
# Step 1: Read camera image (already RGB)
rgb_image = get_image()

# Direct cv image display has bugs on Windows
# Generate random filename (with extension)
file_name = f"random_{uuid.uuid4().hex}.png" # Example output: random_1a2b3c4d5e.png
cv2.imwrite(file_name, rgb_image)
# Create detection task
task = create_task_with_local_image_auto_resize(
api_path="/v2/task/dinox/detection",
api_body_without_image={
"model": "DINO-X-1.0",
"prompt": {
"type": "text",
"text": object_names
},
"targets": ["bbox"],
"bbox_threshold": 0.25,
"iou_threshold": 0.8
},
image_path=file_name
)
client.run_task(task)
result = task.result

# Parse detection results
obj_id_list = [obj['category'] for obj in result['objects']]
obj_locs = [obj['bbox'] for obj in result['objects']]

# Visualization
try:
visualize_result(image_path=file_name, result=result, output_dir="./")
vis_img = Image.open(file_name) # or use output_dir below image
except Exception as e:
vis_img = None
#os.remove(file_name)

return obj_id_list, obj_locs, vis_img


@tool
def ob_objects(obj_name_list:List[str])-> List[Tuple[str, float, float]]:
"""
for drone imagetarget, targetlist [ (object, distance, angle ( with)),...]

Args:
obj_name_list: targetlist, must is, ifinput is in, 

Returns:
List: [(object, and drone distance, and drone angle ( with) >,...]
"""

#step1, object detection
prompt = ".".join(obj_name_list)
# obj_id_list: [obj1, obj2,...], obj_locs: [[xmin, ymin, xmax, ymax],[xmin, ymin, xmax, ymax],...]
obj_id_list, obj_locs = detect(prompt)

#step2, data
responses = client.simGetImages([
# png format
airsim.ImageRequest(0, airsim.ImageType.Scene, pixels_as_float=False, compress=True),

# Floating point uncompressed depth image, pixels represent distance to image plane
airsim.ImageRequest(0, airsim.ImageType.DepthPlanar, pixels_as_float=True),

# Pixels represent distance to camera
airsim.ImageRequest(0, airsim.ImageType.DepthPerspective, pixels_as_float=True)
]
)

img_depth_planar = np.array(responses[1].image_data_float).reshape(responses[1].height, responses[0].width)
img_depth_perspective = np.array(responses[2].image_data_float).reshape(responses[2].height, responses[1].width)

# Regular image
image_data = responses[0].image_data_uint8
img = cv2.imdecode(np.array(bytearray(image_data), dtype='uint8'), cv2.IMREAD_UNCHANGED) # Read from binary image data
img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) # 43


final_obj_list = [] #resultlist
#buildtargetresult
index = 0
for bbox in obj_locs:
center_x = int((bbox[0] + bbox[2]) / 2)
center_y = int((bbox[1] + bbox[3]) / 2)

depth_distance = img_depth_planar[center_y, center_x,] #distance
camera_distance = img_depth_perspective[center_y, center_x] #cameradistance

#angle
angel = math.acos(depth_distance / camera_distance)
angel_degree = math.degrees(angel)

#, edge, edge, yaw angle
if center_x < img.shape[1] / 2:
# iftarget in image, toward, degree 
angel_degree = -1 * angel_degree

obj_name = obj_id_list[index]#target, can has many 

obj_info = (obj_name, camera_distance, angel_degree)
final_obj_list.append(obj_info)
index = index + 1

return final_obj_list

@tool
def watch(prompt:str)->str:
"""
before cameraimage,according toprompt

Args:
prompt: prompt

Return:
str: target use 
"""
# Step 1: Read camera image (already RGB)
rgb_image = get_image()

# Convert to Base64 PNG image
base64_str = cv2_to_base64(rgb_image, ".png") # png or '.jpg'

# Step 2: Image understanding
# Image input:
response = llm_client.chat.completions.create(
model="moonshotai/Kimi-K2.6",
messages=[
{
"role": "user",
"content": [
{"type": "text", "text": prompt},
{
"type": "image_url",
"image_url": {
# "url": "https://yt-shanghai.tos-cn-shanghai.volces.com/tello.jpg"
# Use Base64-encoded local image, note img/png or img/jpg format
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

@tool
def turn(angle: float)->str:
"""
droneangleangle

Args:
angle: droneneed to angle ( with) 

Returns:
str: successstatedescription
"""
yaw_degree = get_yaw()
yaw_degree = yaw_degree + angle
set_yaw(yaw_degree)
return "success"

@tool
def move(distance: float)->str:
"""
toward before distance distance

Args:
distance: drone toward before distance, 

Returns:
str: successstatedescription
"""
step_length = distance
cur_position = get_drone_position()
yaw_degree = cur_position[3]
# Convert angle to radians
yaw = math.radians(yaw_degree)
# Move forward 0.1 meters
x = cur_position[0] + step_length*math.cos(yaw)
y = cur_position[1] + step_length*math.sin(yaw)
z = cur_position[2]
fly_to([x, y, z, 0])
return "success"

if __name__ == "__main__":
takeoff()
object = look()
print(object)
print("done")
