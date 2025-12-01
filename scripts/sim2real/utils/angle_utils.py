import numpy as np


def rad_to_width_m(raw_rad: float, gripper_max_rad: float, gripper_max_m: float) -> float:
        """
        Map from radians (0 rad = fully open, 0.7 rad = fully closed)
        to meters (0.14 m = fully open, 0.0 m = fully closed).

        Args:
            raw_rad (float): Desired gripper angle in radians.
            gripper_max_rad (float): Maximum gripper angle in radians.
            gripper_max_m (float): Maximum gripper width in meters.

        Returns:
            float: Corresponding gripper width in meters.
        """
        # Clamp input angle to the physical range [0, 0.7]
        angle = float(np.clip(raw_rad, 0.0, gripper_max_rad))
        # Linear mapping: inverse relation between angle and width
        width = gripper_max_m * (1.0 - angle / gripper_max_rad)
        # Clamp output width to [0, 0.14]
        return float(np.clip(width, 0.0, gripper_max_m))


def width_m_to_rad(width: float, gripper_max_rad: float, gripper_max_m: float) -> float:
    """
    Map from meters (0.14 m = fully open, 0.0 m = fully closed)
    back to radians (0 rad = fully open, 0.7 rad = fully closed).

    Args:
        width (float): Desired gripper width in meters.

    Returns:
        float: Corresponding gripper angle in radians.
    """
    # Clamp input width to the physical range [0.0, 0.14]
    width = float(np.clip(width, 0.0, gripper_max_m))
    # Linear mapping: inverse relation between width and angle
    angle = gripper_max_rad * (1.0 - width / gripper_max_m)
    # Clamp output angle to [0.0, 0.7]
    return float(np.clip(angle, 0.0, gripper_max_rad))  

def map_joint_angle(pos: float, index: int, arm_sim_dof_angle_limits: list[tuple[int, int, bool]], arm_servo_angle_limits: list[tuple[int, int, bool]]) -> float:
        """
        Map a simulation joint angle (in radians) to the real-world servo angle (in radians).

        Args:
            pos: Joint angle from simulation (in radians).
            index: Index of the joint.
            arm_sim_dof_angle_limits: List of tuples defining the simulation joint limits and inversion.
            arm_servo_angle_limits: List of tuples defining the servo joint limits and inversion.

        Returns:
            Mapped joint angle within the servo limits.
        """
        L, U, inversed = arm_sim_dof_angle_limits[index]
        A, B = arm_servo_angle_limits[index]
        angle_deg = np.rad2deg(float(pos))
        # Check if the simulation angle is within limits
        if not L <= angle_deg <= U:
            # self.get_logger().warn(f"Simulation joint {index} angle ({angle_deg}) out of range [{L}, {U}]. Clipping.")
            angle_deg = np.clip(angle_deg, L, U)
        # Map the angle from the simulation range to the servo range
        mapped = (angle_deg - L) * ((B - A) / (U - L)) + A
        if inversed:
            mapped = (B - A) - (mapped - A) + A
        # Verify the mapped angle is within servo limits
        if not A <= mapped <= B:
            raise Exception(f"Mapped joint {index} angle ({mapped}) out of servo range [{A}, {B}].")
        return mapped
    
def quaternion_angle_error(q1, q2):
    """
    Compute angular error (in radians) between two unit quaternions.
    Quaternions must be in [w, x, y, z] convention.

    Args:
        q1: First quaternion.
        q2: Second quaternion.
    Returns:
        Angular error in radians.
    """
    q1 = np.array(q1, dtype=float)
    q2 = np.array(q2, dtype=float)

    # normalize
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)

    # dot product
    dot = np.dot(q1, q2)

    # clamp
    dot = np.clip(dot, -1.0, 1.0)

    # angular error
    theta = 2 * np.arccos(abs(dot))
    return theta