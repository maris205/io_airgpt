## Project Overview  
The project is named **IO.AIRGPT: Flying a Drone with Large Language Model**.  
This project uses **IO Intelligence** to control drones based on natural language instructions, enabling autonomous flight in the **AirSim simulation environment**.

<img src="img/cover.png" alt="Project Design" width="600">

---

## Introduction to IO Intelligence  
IO Intelligence features advanced economic model design that allows compute providers to earn rewards during idle time, fundamentally ensuring cost-effectiveness and service quality of computing power. This provides developers of large language model applications with a new and competitive option.

---

## Introduction to AirSim  
AirSim is an open-source drone and autonomous vehicle simulator designed to provide a high-fidelity simulation environment for research and development of drones and autonomous vehicles. Developed by Microsoft, it supports a variety of sensor simulations such as cameras and LiDAR, and can interface with multiple programming languages like Python and C++.

---

## Project Architecture  
Our project combines **IO Intelligence** and **AirSim**, leveraging the powerful reasoning capabilities of large language models from IO to enable autonomous drone flight within the AirSim environment.

The system adopts a typical **embodied intelligence framework**, where IO provides key planning, perception, and decision-making capabilities.

<img src="img/workflow.png" alt="IO.AIRGPT Framework" width="600">

---

## Project Execution

- `1-airsim_io_env.ipynb`: Basic environment setup, including IO and AirSim
- `2-basic_control.ipynb`: Basic drone control
- `3-complex_control.ipynb`: Advanced drone control
- `4-solar_matrix.ipynb`: Full drone mission execution
- `airsim_agent.py`: Encapsulation of IO Intelligence large model service (please fill in your own API key)
- `airsim_wrapper.py`: Encapsulation of AirSim, optimized for LLM calls

**Scene 1:**  
<img src="img/s1.png" alt="IO.AIRGPT Framework" width="600">

**Scene 2:**  
<img src="img/s2.png" alt="IO.AIRGPT Framework" width="600">

**Scene 3:**  
<img src="img/s3.png" alt="IO.AIRGPT Framework" width="600">

---

## Next Steps  

In the future, we will build upon IO's cloud services and IO Intelligence to:
- Implement training and deployment of drone-focused large models and inference services.
- Leverage IO Cloud’s Unreal Engine integration to run AirSim in the cloud, completing the technical loop.
- Integrate payment functionality using IO Tokens.

Ultimately, our goal is to create a **complete drone large model learning and development platform**.

**Future System Design:**  
<img src="img/system.png" alt="Platform Architecture" width="600">

**Future Product Design:**  
<img src="img/product.png" alt="Product Design" width="600">