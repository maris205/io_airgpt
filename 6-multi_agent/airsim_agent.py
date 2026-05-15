import sys
sys.path.append('..')

import airsim
from langchain_core.tools import tool
from typing import Dict

# Initialize AirSim client
client = airsim.MultirotorClient()
client.confirmConnection()

@tool
def takeoff(drone_id: str) -> str:
    """
    Command the drone to take off.
    :param drone_id: The drone to control (e.g., 'Drone1').
    """
    print(f"Executing: takeoff for {drone_id}")
    # Must enable API control first
    client.enableApiControl(True, vehicle_name=drone_id)
    client.armDisarm(True, vehicle_name=drone_id)
    # Async takeoff, wait for completion
    client.takeoffAsync(vehicle_name=drone_id).join()
    return f"Drone {drone_id} has taken off successfully."

@tool
def fly_to_position(drone_id: str, x: float, y: float, z: float) -> str:
    """
    Command the drone to fly to a position at 5 m/s velocity in NED coordinates.
    Z is negative for above ground (higher altitude = more negative Z).
    :param drone_id: The drone to control.
    :param x: Target position X coordinate.
    :param y: Target position Y coordinate.
    :param z: Target position Z coordinate (negative = higher altitude).
    """
    print(f"Executing: fly_to_position for {drone_id} to ({x}, {y}, {z})")
    client.moveToPositionAsync(x, y, z, 5, vehicle_name=drone_id).join()
    return f"Drone {drone_id} has arrived at target position ({x}, {y}, {z})."

@tool
def get_drone_state(drone_id: str) -> Dict:
    """
    Get the drone's current state, including position and orientation.
    :param drone_id: The drone to query.
    """
    print(f"Executing: get_drone_state for {drone_id}")
    state = client.getMultirotorState(vehicle_name=drone_id)
    # Return dictionary for LLM processing
    return {
        "position": {
            "x_val": state.kinematics_estimated.position.x_val,
            "y_val": state.kinematics_estimated.position.y_val,
            "z_val": state.kinematics_estimated.position.z_val,
        },
        "orientation": {
            "w_val": state.kinematics_estimated.orientation.w_val,
            "x_val": state.kinematics_estimated.orientation.x_val,
            "y_val": state.kinematics_estimated.orientation.y_val,
            "z_val": state.kinematics_estimated.orientation.z_val,
        }
    }
