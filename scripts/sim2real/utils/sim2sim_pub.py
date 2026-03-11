import omni.isaac.core.utils.extensions as extensions
# Force load extensions to prevent missing module errors
extensions.enable_extension("isaacsim.ros2.bridge")
extensions.enable_extension("omni.isaac.core")

import omni.graph.core as og
import omni.usd
import omni.kit.commands
import omni.physx
from pxr import Sdf, UsdPhysics, PhysxSchema, Gf
from scipy.spatial.transform import Rotation as R
import numpy as np

# ROS Imports
try:
    import rclpy
    from geometry_msgs.msg import WrenchStamped
except ImportError:
    pass

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.world import World

# --- CONFIGURATION ---
ROBOT_PATH = "/World/Rizon4s_with_Grav" 
GRAPH_PATH = "/ActionGraph"
ROBOT_SN = "sn" 
TOPIC_WRENCH_WORLD = f"/{ROBOT_SN}/external_wrench_in_world"
SENSOR_JOINT_NAME = "flange_to_gripper"

# --- STATE MANAGEMENT ---
class WrenchPublisherState:
    def __init__(self):
        self.node = None
        self.pub = None
        self.sub = None
        self.robot = None
        self.sensor_joint_index = -1
        self.sensor_prim_path = ""
        self.last_error = ""

if not hasattr(omni, "wrench_pub_state"):
    omni.wrench_pub_state = WrenchPublisherState()
state = omni.wrench_pub_state

# --- HELPERS ---
def setup_force_sensor(robot_path):
    stage = omni.usd.get_context().get_stage()
    joint_prim_path = f"{robot_path}/joints/{SENSOR_JOINT_NAME}"
    joint_prim = stage.GetPrimAtPath(joint_prim_path)
    
    if not joint_prim.IsValid():
        print(f"[WARNING] '{SENSOR_JOINT_NAME}' not found. Falling back to 'joint7'.")
        joint_prim_path = f"{robot_path}/joints/joint7"
        joint_prim = stage.GetPrimAtPath(joint_prim_path)
    
    state.sensor_prim_path = joint_prim_path 

    omni.kit.commands.execute(
        "AddPhysicsComponent",
        usd_prim=joint_prim, 
        component="PhysxArticulationForceSensorAPI"
    )

def create_standard_graph():
    keys = og.Controller.Keys
    if get_prim_at_path(GRAPH_PATH):
        return

    og.Controller.edit(
        {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
            ],
            keys.SET_VALUES: [
                ("SubscribeJointState.inputs:topicName", "/joint_command"),
            ],
        },
    )
    
    def set_target(node, rel, target):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(node)
        rel = prim.CreateRelationship(rel, custom=False)
        rel.SetTargets([Sdf.Path(target)])

    set_target(f"{GRAPH_PATH}/ArticulationController", "inputs:targetPrim", ROBOT_PATH)
    set_target(f"{GRAPH_PATH}/PublishJointState", "inputs:targetPrim", ROBOT_PATH)
    set_target(f"{GRAPH_PATH}/PublishTF", "inputs:targetPrims", ROBOT_PATH)

# --- PHYSICS CALLBACK ---
def on_physics_step(dt):
    if not state.robot: return

    # Self-Healing
    if not state.robot.handles_initialized:
        if state.robot.prim.IsValid():
            try: state.robot.initialize()
            except: return 
        else: return 

    try:
        # 1. Get Raw Data
        forces = state.robot.get_measured_joint_forces()
        if forces is None: return

        # 2. Find Sensor Index
        if state.sensor_joint_index == -1:
            dof_names = state.robot.dof_names
            found = False
            for i, name in enumerate(dof_names):
                if SENSOR_JOINT_NAME in name:
                    state.sensor_joint_index = i
                    found = True
                    break
            if not found:
                state.sensor_joint_index = len(forces) - 1

        # 3. Extract Local Wrench
        raw_wrench = forces[state.sensor_joint_index, :] 
        force_local = raw_wrench[0:3]
        torque_local = raw_wrench[3:6]

        # 4. ROTATION FIX
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(state.sensor_prim_path)
        
        # Get Transform directly (It returns Gf.Matrix4d)
        transform_matrix = omni.usd.get_world_transform_matrix(prim)
        
        # Extract Rotation directly from the Matrix
        rotation = transform_matrix.ExtractRotation()
        quat = rotation.GetQuat() # (w, x, y, z)
        
        # Convert to Scipy format (x, y, z, w)
        imag = quat.GetImaginary()
        scipy_quat = [imag[0], imag[1], imag[2], quat.GetReal()]
        
        # Apply Rotation
        r = R.from_quat(scipy_quat)
        force_world = r.apply(force_local)
        torque_world = r.apply(torque_local)

        # 5. Publish
        msg = WrenchStamped()
        msg.header.stamp = state.node.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        
        msg.wrench.force.x = float(force_world[0])
        msg.wrench.force.y = float(force_world[1])
        msg.wrench.force.z = float(force_world[2])
        msg.wrench.torque.x = float(torque_world[0])
        msg.wrench.torque.y = float(torque_world[1])
        msg.wrench.torque.z = float(torque_world[2])
        
        state.pub.publish(msg)
        
    except Exception as e:
        if state.last_error != str(e):
            print(f"[ERROR in Callback] {e}")
            state.last_error = str(e)

# --- EXECUTION ---
setup_force_sensor(ROBOT_PATH)
create_standard_graph()

world = World.instance()
if world is None:
    world = World()

try:
    rclpy.init()
except:
    pass

if state.node is None:
    state.node = rclpy.create_node("isaac_wrench_script_pub")
    state.pub = state.node.create_publisher(WrenchStamped, TOPIC_WRENCH_WORLD, 10)
    
    state.robot = world.scene.add(Articulation(ROBOT_PATH, name="rizon_robot"))
    
    print("[INFO] Resetting World to initialize physics handles...")
    world.reset()

    physx_interface = omni.physx.get_physx_interface()
    if state.sub: state.sub = None 
    state.sub = physx_interface.subscribe_physics_step_events(on_physics_step)
    
    print(f"[SUCCESS] Wrench Publishing to {TOPIC_WRENCH_WORLD} (World Frame)")
else:
    print("[INFO] Logic updated.")
    physx_interface = omni.physx.get_physx_interface()
    state.sub = physx_interface.subscribe_physics_step_events(on_physics_step)