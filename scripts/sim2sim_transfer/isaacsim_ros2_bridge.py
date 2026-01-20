# FILE: isaacsim_ros2_bridge.py

import omni.isaac.core.utils.extensions as extensions
# Abilitiamo l'estensione "Nucleare" che usa Isaac Lab sotto il cofano
extensions.enable_extension("omni.physx.tensors")

import omni.graph.core as og
import omni.physx
import omni.timeline
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

# Importiamo l'API diretta dei tensori
import omni.physics.tensors as physx_tensors

try:
    import rclpy
    from geometry_msgs.msg import WrenchStamped
except ImportError:
    pass

from pxr import Sdf
from omni.isaac.core.world import World
from omni.isaac.core.utils.prims import get_prim_at_path

# --- CONFIGURAZIONE ---
ROBOT_PATH = "/World/Rizon4s_with_Grav" 
GRAPH_PATH = "/ActionGraph"
ROBOT_SN = "sn" 
TOPIC_WRENCH_WORLD = f"/{ROBOT_SN}/external_wrench_in_world"
TARGET_BODY_NAME = "flange" 

# --- STATO ---
class WrenchPublisherState:
    def __init__(self):
        self.node = None
        self.pub = None
        self.sub = None
        self.sim_view = None  # QUESTO sarà il nostro "root_physx_view"
        self.body_idx = None
        self.init_done = False

if not hasattr(omni, "wrench_pub_state"):
    omni.wrench_pub_state = WrenchPublisherState()
state = omni.wrench_pub_state

# --- CALLBACK ---
def on_physics_step(dt):
    timeline = omni.timeline.get_timeline_interface()
    # I tensori funzionano solo se la simulazione sta girando
    if not timeline.is_playing(): return

    # 1. INIZIALIZZAZIONE (Manuale, senza classi wrapper)
    if not state.init_done:
        try:
            print("[INIT] Creazione diretta della Simulation View (Root PhysX View)...")
            
            # Questa chiamata crea l'oggetto che nel training chiamano 'root_physx_view'
            # backend="torch" ci dà tensori GPU diretti
            state.sim_view = physx_tensors.create_simulation_view("torch")
            
            # Diciamo alla view di guardare solo il nostro robot
            state.sim_view.set_subspace_roots([ROBOT_PATH])
            
            print("[SUCCESS] View Tensoriale creata.")
            
            # --- MAPPARE GLI INDICI ---
            # La view tensoriale lavora con indici piatti. Dobbiamo trovare quale indice
            # corrisponde alla flangia.
            # get_rigid_body_names() restituisce i path completi o parziali
            body_paths = state.sim_view.get_rigid_body_names()
            
            # Cerchiamo l'indice che contiene "flange"
            found = False
            for i, path in enumerate(body_paths):
                if TARGET_BODY_NAME in path:
                    state.body_idx = i
                    print(f"[INDEX] Trovato target '{TARGET_BODY_NAME}' all'indice {i}")
                    print(f"        Path completo: {path}")
                    found = True
                    break
            
            if not found:
                print(f"[ERROR] Non ho trovato '{TARGET_BODY_NAME}' nei body della view.")
                print(f"        Body disponibili: {body_paths}")
                return

            state.init_done = True
        except Exception as e:
            # Spesso fallisce al primo frame se PhysX non è "caldo", riprova silenziosamente
            # print(f"[INIT WAIT] {e}") 
            return

    # 2. LETTURA DATI (Esattamente come nel Training)
    try:
        if state.sim_view is None: return

        # ECCOLA: La chiamata che usano nel training
        # Restituisce [Num_Envs, Num_Links, 6]
        forces = state.sim_view.get_link_incoming_joint_force()
        
        # Estrazione (Env 0)
        f_tensor = forces[0, state.body_idx, 0:3] 
        t_tensor = forces[0, state.body_idx, 3:6]
        
        # --- ROTAZIONE ---
        # Per coerenza, prendiamo anche le pose dalla stessa view tensoriale
        transforms = state.sim_view.get_rigid_body_poses()
        # transforms shape: [Num_Envs, Num_Bodies, 7] (Pos: 0-2, Rot: 3-6)
        
        # Isaac Sim Core (PhysX) usa quaternioni (x, y, z, w) o (w, x, y, z)?
        # I tensori PhysX puri di solito usano (x, y, z, w).
        # Verifichiamo la magnitudo per sicurezza.
        quat = transforms[0, state.body_idx, 3:7]
        
        q = quat.cpu().numpy()
        f = f_tensor.cpu().numpy()
        t = t_tensor.cpu().numpy()
        
        # Gestione NaN (capita se la sim esplode o è ferma)
        if np.isnan(f).any(): return

        # Scipy usa (x, y, z, w).
        # Tentativo standard: assumiamo input (x, y, z, w) dai tensori PhysX
        r = R.from_quat([q[0], q[1], q[2], q[3]]) 
        
        # Se i valori sembrano strani (rotazione sbagliata), prova l'ordine w,x,y,z:
        # r = R.from_quat([q[1], q[2], q[3], q[0]])

        f_world = r.apply(f)
        t_world = r.apply(t)

        # STAMPA DEBUG (Ogni tanto)
        f_mag = np.linalg.norm(f_world)
        if np.random.rand() < 0.05: # 5% dei frame
            print(f"[TENSOR] Fz: {f_world[2]:.3f} | Mag: {f_mag:.3f}")

        # Publish ROS
        msg = WrenchStamped()
        msg.header.stamp = state.node.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.wrench.force.x = float(f_world[0])
        msg.wrench.force.y = float(f_world[1])
        msg.wrench.force.z = float(f_world[2])
        msg.wrench.torque.x = float(t_world[0])
        msg.wrench.torque.y = float(t_world[1])
        msg.wrench.torque.z = float(t_world[2])
        state.pub.publish(msg)

    except Exception as e:
        # print(f"[RUNTIME] {e}")
        pass

# --- SETUP GRAFO (Standard) ---
def create_graph():
    keys = og.Controller.Keys
    if get_prim_at_path(GRAPH_PATH): return
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
            keys.SET_VALUES: [("SubscribeJointState.inputs:topicName", "/joint_command")],
        },
    )
    stage = omni.usd.get_context().get_stage()
    for node in ["ArticulationController", "PublishJointState"]:
        prim = stage.GetPrimAtPath(f"{GRAPH_PATH}/{node}")
        rel = prim.CreateRelationship("inputs:targetPrim", custom=False)
        rel.SetTargets([Sdf.Path(ROBOT_PATH)])
    prim = stage.GetPrimAtPath(f"{GRAPH_PATH}/PublishTF")
    rel = prim.CreateRelationship("inputs:targetPrims", custom=False)
    rel.SetTargets([Sdf.Path(ROBOT_PATH)])

world = World.instance()
if world is None: world = World()

# CLEANUP
if state.sub: 
    state.sub.unsubscribe()
    state.sub = None
state.sim_view = None
state.init_done = False

try: rclpy.init()
except: pass
if state.node is None:
    state.node = rclpy.create_node("isaac_wrench_script_pub")
    state.pub = state.node.create_publisher(WrenchStamped, TOPIC_WRENCH_WORLD, 10)

create_graph()

if get_prim_at_path(ROBOT_PATH).IsValid():
    print("[INFO] Resetting World...")
    world.reset()
    if world.scene.get_object("rizon_robot"):
        world.scene.remove_object("rizon_robot")

    physx_interface = omni.physx.get_physx_interface()
    state.sub = physx_interface.subscribe_physics_step_events(on_physics_step)
    print(f"[SUCCESS] Script v39 (Direct Root View) Caricato.")
    print("Bypass totale dei wrapper Python. Accesso diretto alla GPU.")
else:
    print(f"[FATAL] Robot non trovato: {ROBOT_PATH}")