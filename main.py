import math
import mujoco
import mujoco.viewer
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import python.a_star as astar
import python.cmpe434_dungeon as dungeon
from python.config import *
from python.controller import Controller
from python.utils import *

# Helper construsts for the viewer for pause/unpause functionality.
paused = False

# Pressing SPACE key toggles the paused state.
def mujoco_viewer_callback(keycode):
    global paused
    if keycode == ord(' '):  # Use ord(' ') for space key comparison
        paused = not paused

class Robot:
    def __init__(self, d, robot_id):
        self.d = d
        self.robot_id = robot_id
    def position(self):
        return self.d.xpos[self.robot_id]
    def quat(self):
        return self.d.qpos[3:7]
    def yaw(self):
        return quat_to_yaw(self.quat())

def main():
    # Load existing XML models
    scene_spec = mujoco.MjSpec.from_file("scenes/empty_floor.xml")

    tiles, rooms, connections = dungeon.generate(3, 2, 8)
    for index, r in enumerate(rooms):
        (xmin, ymin, xmax, ymax) = dungeon.find_room_corners(r)
        scene_spec.worldbody.add_geom(name='R{}'.format(index), type=mujoco.mjtGeom.mjGEOM_PLANE, size=[(xmax-xmin)+1, (ymax-ymin)+1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[(xmin+xmax), (ymin+ymax), 0])

    for pos, tile in tiles.items():
        if tile == "#":
            scene_spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[1, 1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[pos[0]*2, pos[1]*2, 0])

    start_pos = random.choice([key for key in tiles.keys() if tiles[key] == "."])
    final_pos = random.choice([key for key in tiles.keys() if tiles[key] == "." and key != start_pos])

    #Extract wall, start and finish positions for path planning and avoidance
    wall_positions = [pos for pos, t in tiles.items() if t == "#"]
    ox, oy = zip(*[pos for pos, t in tiles.items() if t == "#"])
    sx, sy = start_pos
    gx, gy = final_pos

    #Plan the path
    planner = astar.AStarPlanner(ox, oy, resolution=1.0, rr=0.5)
    rx, ry = planner.planning(sx, sy, gx, gy)

    #Make the path denser, smoother and turn friendly
    raw_path = list(zip(rx, ry))
    smooth_path = insert_via_points(raw_path, tiles)
    adjusted_path = simplify_path(smooth_path, tiles)
    path = densify_path(adjusted_path, 0.25)

    rx, ry = zip(*path)

    #Plot the found path
    plt.figure()
    plt.scatter(ox, oy, marker="s")          # walls
    plt.scatter([sx], [sy], marker="o")      # start
    plt.scatter([gx], [gy], marker="x")      # goal
    plt.plot(rx, ry)                         # computed path
    plt.gca().set_aspect("equal", "box")
    plt.title("Found Path")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

    #Convert the planned path to actual simulation path
    rx = [2 * x for x in rx]
    ry = [2 * y for y in ry]
    path_coords = list(zip(rx, ry))
    path_coords = path_coords[::-1]

    if DEBUG:
        print("Planned path waypoints (x, y):")
        for idx, (x, y) in enumerate(path_coords):
            print(f"  {idx:2d}: ({x:.2f}, {y:.2f})")

    scene_spec.worldbody.add_site(name='start', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, 0.5, 0.01], rgba=[0, 0, 1, 1],  pos=[start_pos[0]*2, start_pos[1]*2, 0])
    scene_spec.worldbody.add_site(name='finish', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, 0.5, 0.01], rgba=[1, 0, 0, 1],  pos=[final_pos[0]*2, final_pos[1]*2, 0])
    robot_spec = mujoco.MjSpec.from_file("models/mushr_car/model.xml")

    # Add robots to the scene:
    # - There must be a frame or site in the scene model to attach the robot to.
    # - A prefix is required if we add multiple robots using the same model.
    scene_spec.attach(robot_spec, frame="world", prefix="robot-")
    scene_spec.body("robot-buddy").pos[0] = start_pos[0] * 2
    scene_spec.body("robot-buddy").pos[1] = start_pos[1] * 2

    # Randomize initial orientation
    yaw = np.random.uniform(-np.pi, np.pi)
    euler = np.array([0.0, 0.0, yaw], dtype=np.float64)
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_euler2Quat(quat, euler, 'xyz')
    scene_spec.body("robot-buddy").quat[:] = quat

    # Add obstacles to the scene
    for i, room in enumerate(rooms):
        obs_pos = random.choice([tile for tile in room if tile != start_pos and tile != final_pos])
        scene_spec.worldbody.add_geom(
            name='Z{}'.format(i), 
            type=mujoco.mjtGeom.mjGEOM_CYLINDER, 
            size=[0.2, 0.05, 0.1], 
            rgba=[0.8, 0.0, 0.1, 1],  
            pos=[obs_pos[0]*2, obs_pos[1]*2, 0.08]
        )

    # Add visuals to the waypoints
    for i, room in enumerate(path_coords):
        scene_spec.worldbody.add_geom(
            name='L{}'.format(i), 
            type=mujoco.mjtGeom.mjGEOM_BOX, 
            size=[0.05, 0.05, 0.001], 
            rgba=[0.8, 0.0, 0.1, 1],  
            pos=[room[0], room[1], 0.001]
        )

    # Initalize our simulation
    # Roughly, m keeps static (model) information, and d keeps dynamic (state) information. 
    m = scene_spec.compile()
    d = mujoco.MjData(m)

    obstacles = [m.geom(i).id for i in range(m.ngeom) if m.geom(i).name.startswith("Z")]
    uniform_direction_dist = sp.stats.uniform_direction(2)
    obstacle_direction = [[x, y, 0] for x,y in uniform_direction_dist.rvs(len(obstacles))]
    unused = np.zeros(1, dtype=np.int32)

    aligned = True
    reverse = False
    #Find desired heading
    if len(path_coords) >= 2:
        start_point = np.array([start_pos[0] * 2, start_pos[1] * 2])
        wp1 = np.array(path_coords[1])
        
        desired_quat = get_heading_quaternion(start_point, wp1)
        aligned = False

    #Pure pursuit controller
    controller = Controller(path_coords, LOOKAHEAD_DISTANCE, gain=P_GAIN, throttle=THROTTLE)
    previous_velocity = 0.0
    previous_steering = 0.0

    with mujoco.viewer.launch_passive(m, d, key_callback=mujoco_viewer_callback) as viewer:

      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
      viewer.cam.fixedcamid = m.camera("robot-third_person").id

      # These actuator names are defined in the model XML file for the robot.
      # Prefixes distinguish from other actuators from the same model.
      velocity = d.actuator("robot-throttle_velocity")
      steering = d.actuator("robot-steering")

      #Robot data
      robot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot-buddy")
      robot = Robot(d, robot_id)

      # Close the viewer automatically after 30 wall-clock-seconds.
      start = time.time()
      while viewer.is_running() and time.time() - start < 3000:
        step_start = time.time()

        #If initial heading is not to the first waypoint, turn the robot while avoiding obstacles and walls
        if not aligned:
            relative_quat = quat_multiplication(desired_quat, quat_conj(robot.quat()))
            yaw_error = rad_to_max_pi(quat_to_yaw(relative_quat))
            if DEBUG:
                print(f"Yaw error = {math.degrees(yaw_error):.1f}°")
            repulsion_wall = compute_repulsion_force(robot.position(), wall_positions, repulsion_radius=WALL_REPULSION_RADIUS, strength=WALL_REPULSION_STRENGTH_ALIGN)
            repulsion_obstacle = compute_obstacle_repulsion(robot.position(), robot.yaw(), m, obstacles, repulsion_radius=OBSTACLE_REPULSION_RADIUS, strength=OBSTACLE_REPULSION_STRENGTH_ALIGN)
            repulsion_force = repulsion_wall + repulsion_obstacle
            if DEBUG:
                print(f"Repulsion force: {repulsion_force}, mag: {np.linalg.norm(repulsion_force)}")

            if not reverse and np.linalg.norm(repulsion_force) > REPULSION_BACK_THRESHOLD_ALIGN:
                reverse = True
            elif reverse and np.linalg.norm(repulsion_force) < REPULSION_SAFE_THRESHOLD:
                reverse = False

            if np.linalg.norm(repulsion_force) > 0:
                heading_vector = np.array([np.cos(robot.yaw()), np.sin(robot.yaw())])
                adjusted_vector = heading_vector + repulsion_force
                adjusted_yaw = np.arctan2(adjusted_vector[1], adjusted_vector[0])
                adjusted_yaw_error = rad_to_max_pi(adjusted_yaw - robot.yaw())
                steer_cmd = np.clip(adjusted_yaw_error * ALIGN_REPULSION_GAIN, -MAX_STEER_VAL_ALIGN, MAX_STEER_VAL_ALIGN)
            else:
                steer_cmd = math.copysign(MAX_STEER_VAL_ALIGN, yaw_error)

            if abs(yaw_error) < YAW_THRESHOLD:
                aligned = True
                if DEBUG:
                    print("Aligned")
            velocity.ctrl = -ALIGN_THROTTLE if reverse else ALIGN_THROTTLE
            steering.ctrl = steer_cmd
        else:
            #While driving through waypoints check for obstacles, avoid them, if too close to a wall go back until
            #have a safe distance (to not hit the walls)
            base_steer = controller.control_command(robot.position()[:2], robot.yaw())
            repulsion_wall = compute_repulsion_force(robot.position(), wall_positions, repulsion_radius=WALL_REPULSION_RADIUS, strength=WALL_REPULSION_STRENGTH)
            repulsion_obstacle = compute_obstacle_repulsion(robot.position(), robot.yaw(), m, obstacles, repulsion_radius=OBSTACLE_REPULSION_RADIUS, strength=OBSTACLE_REPULSION_STRENGTH)
            repulsion_force = repulsion_wall + repulsion_obstacle
            if DEBUG:
                print(f"Repulsion force: {repulsion_force}, mag: {np.linalg.norm(repulsion_force)}")

            if not reverse and np.linalg.norm(repulsion_wall) > REPULSION_BACK_THRESHOLD:
                reverse = True
            elif reverse and np.linalg.norm(repulsion_wall) < REPULSION_SAFE_THRESHOLD:
                reverse = False

            if np.linalg.norm(repulsion_force) > 0:
                direction_vector = np.array([np.cos(robot.yaw()), np.sin(robot.yaw())])
                adjusted_direction = direction_vector + repulsion_force
                adjusted_yaw = np.arctan2(adjusted_direction[1], adjusted_direction[0])
                yaw_error = rad_to_max_pi(adjusted_yaw - robot.yaw())
                avoid_steer = np.clip(yaw_error * REPULSION_GAIN, -MAX_STEER_VAL, MAX_STEER_VAL)
                steering.ctrl = avoid_steer
            else:
                steering.ctrl = base_steer

            if np.linalg.norm(robot.position()[:2] - path_coords[-1]) < WP_THRESHOLD:
                velocity.ctrl = 0.0
                if DEBUG:
                    print("Goal reached!")
            else:
                velocity.ctrl = -BACK_THROTTLE if reverse else controller.throttle

        #Apply filter for smoother driving
        velocity.ctrl = alpha * velocity.ctrl + (1 - alpha) * previous_velocity
        steering.ctrl = alpha * steering.ctrl + (1 - alpha) * previous_steering
        if abs(steering.ctrl) < 0.2:
            steering.ctrl = 0.0

        previous_velocity = velocity.ctrl
        previous_steering = steering.ctrl     

        # Update obstables (bouncing movement)
        for i, x in enumerate(obstacles):
            dx = obstacle_direction[i][0]
            dy = obstacle_direction[i][1]

            px = m.geom_pos[x][0]
            py = m.geom_pos[x][1]
            pz = 0.02

            nearest_dist = mujoco.mj_ray(m, d, [px, py, pz], obstacle_direction[i], None, 1, -1, unused)

            if nearest_dist >= 0 and nearest_dist < 0.4:
                obstacle_direction[i][0] = -dy
                obstacle_direction[i][1] = dx

            m.geom_pos[x][0] = m.geom_pos[x][0]+dx*0.001
            m.geom_pos[x][1] = m.geom_pos[x][1]+dy*0.001
        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(m, d)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
          time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
