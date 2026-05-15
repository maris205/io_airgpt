import sys
sys.path.append('..')
import airsim
import math
import numpy as np
import cv2
import base64
import os
from openai import OpenAI
from PIL import Image
import uuid
from dds_cloudapi_sdk.tasks.v2_task import create_task_with_local_image_auto_resize
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.visualization_util import visualize_result

api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # Use your own API key from io.solutions
gdino_token = "xxxxxxxxxxxxxxxxxxxxxxxxx" # Use your own token for DINO

objects_dict = {
"cola": "airsim_coca",
"orchid": "airsim_lanhua",
"coconut water": "airsim_yezishui",
"rubber duck": "airsim_duck",
"mirror": "airsim_mirror_06",
"square table": "airsim_fangzhuo",
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
object_names_ue = []
while len(object_names_ue) == 0:
object_names_ue = self.client.simListSceneObjects(query_string)
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
# Encode as byte stream
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
# Step 1: Read camera image (already RGB)
rgb_image = self.get_image()

# Convert to Base64 PNG image
base64_str = self.cv2_to_base64(rgb_image, ".png") # png or '.jpg'

# Step 2: Image understanding
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

def detect(self, object_names):
"""
Perform object detection on a local image, returning detected categories, bounding boxes, and visualization
:param object_names: Detection targets (comma-separated English string, e.g., 'duck, cola')
:return: obj_id_list, obj_locs, vis_img
"""
config = Config(gdino_token)
client = Client(config)
# Step 1: Read camera image (already RGB)
rgb_image = self.get_image()

# Direct cv image display has bugs on Windows
# Generate random filename (with extension)
file_name = f"random_{uuid.uuid4().hex}.png" # Example: random_1a2b3c4d5e.png
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
