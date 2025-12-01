# Sim2Real Script

## Overview

This directory contains scripts for sim-to-real tasks, including reaching and lifting tasks for robotic manipulation. The scripts are designed to interface with a robotic system and execute predefined policies.

## Prerequisites

To run the scripts, ensure the following requirements are met:

1. **Robotiq URCap Adapter**  
   Install the Robotiq URCap Adapter from the following repository:  
   [https://github.com/fzi-forschungszentrum-informatik/robotiq_2f_urcap_adapter/tree/main/action](https://github.com/fzi-forschungszentrum-informatik/robotiq_2f_urcap_adapter/tree/main/action)

2. **Policies**  
   Import the required policies under the `sim2real/policies` directory. These policies define the behavior for the tasks.

3. **Ethernet Connection**  
   Ensure that your system is connected to the UR10e robot through an Ethernet connection.

## Usage

- **Reach Task**  
  Run the `run_reach_task.py` script to execute the reach task:  
  ```bash
  python scripts/sim2real/run_reach_task.py
  ```

- **Lift Task**  
  Run the `run_lift_task.py` script to execute the lift task:  
  ```bash
  python scripts/sim2real/run_lift_task.py
  ```
