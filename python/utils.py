import mujoco
import numpy as np
import math

def find_intersection(position, lookahead_distance, wp1, wp2):
    """Find intersection between look ahead circle and trajectory
    @param position robot position
    @param lookahead_distance look ahead parameter
    @param wp1 first trajectory point
    @param wp2 second trajectory point
    @return list of intersections
    """
    direction = wp2 - wp1 
    center2wp1 = wp1 - position

    a = np.dot(direction, direction)
    b = 2 * np.dot(center2wp1, direction)
    c = np.dot(center2wp1, center2wp1) - lookahead_distance ** 2

    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return []

    sol1 = (-b + np.sqrt(discriminant)) / (2 * a)
    sol2 = (-b - np.sqrt(discriminant)) / (2 * a)

    intersections = []
    if 0 <= sol1 <= 1:
        intersections.append(wp1 + sol1 * direction)
    if 0 <= sol2 <= 1:
        intersections.append(wp1 + sol2 * direction)
        
    return intersections

def insert_via_points(path, tiles):
    """Insert additional points so that the path has 90 degrees corners
    @param path (x, y) of waypoints from planner
    @param tiles dungeon generation tiles
    @return path with via points for right angle corners
    """
    free_tiles = {pos for pos, val in tiles.items() if val == "."}
    new_path = []
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        new_path.append((x0, y0))

        if x0 != x1 and y0 != y1:
            tile1 = (x0, y1)
            tile2 = (x1, y0)
            if tile1 in free_tiles:
                new_path.append(tile1)
            elif tile2 in free_tiles:
                new_path.append(tile2)
    new_path.append(path[-1])

    return new_path

def densify_path(path, max_spacing):
    """Insert intermediate points between path points with specified spacing
    @param path (x, y) of waypoints from planner
    @param max_spacing maximum distance between points
    @return path with denser waypoints
    """
    densified = []
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        dx = x1 - x0
        dy = y1 - y0
        n = max(1, int(math.ceil(math.hypot(dx, dy) / max_spacing)))
        for i in range(n):
            spacing = i / n
            densified.append((x0 + spacing * dx, y0 + spacing * dy))
    densified.append(path[-1])

    return densified

def compute_repulsion_force(position, wall_positions, repulsion_radius=1.0, strength=2.0):
    """Compute a 2D repulsion vector from nearby walls.
    @param position position of robot
    @param wall_positions positions of wall center points
    @param repulsion_radius radius of obstacle repulsion distance
    @param strength repulsion strength
    @return force"""
    force = np.zeros(2)
    for wx, wy in wall_positions:
        wall_center = np.array([wx * 2, wy * 2])
        dist = np.linalg.norm(position[:2] - wall_center)
        if 0 < dist < repulsion_radius:
            direction = (position[:2] - wall_center) / dist
            magnitude = strength * (1.0 / dist - 1.0 / repulsion_radius)
            force += direction * magnitude
    return force

def compute_obstacle_repulsion(position, heading, m, obstacle_ids,
                               repulsion_radius=1.0, strength=2.0):
    """Compute a 2D repulsion vector from nearby obstacles.
    @param position position of robot
    @param heading heading of robot
    @param m model
    @param obstacle_ids ids of obstacles
    @param repulsion_radius radius of obstacle repulsion distance
    @param strength repulsion strength
    @return force"""
    force = np.zeros(2)
    heading_vector = np.array([math.cos(heading), math.sin(heading)])
    for obs_id in obstacle_ids:
        obstacle_position = m.geom_pos[obs_id][:2]
        direction_vector = obstacle_position - position[:2]
        distance = np.linalg.norm(direction_vector)

        if distance == 0 or distance > repulsion_radius:
            continue

        if np.dot(direction_vector, heading_vector) < 0:
            continue

        frontness = np.dot(direction_vector, heading_vector) / distance
        if frontness <= 0:
            continue  

        cos_theta = np.dot(heading_vector, direction_vector)  
        angle_gain = 1.0 + 2.0 * (cos_theta ** 4)

        direction = -direction_vector/distance                
        magnitude = (1 / distance ** 2 - 1 / repulsion_radius ** 2)
        magnitude = strength * frontness * max(0, magnitude) * angle_gain
        force += direction * magnitude

    return force

def get_heading_quaternion(init_pos, desired_pos):
    """Calculate quaternion to point from initial to desired position
    @param init_pos (x,y) of starting position
    @param desired_pos (x,y) of target position
    @return quaternion 
    """
    dx = desired_pos[0] - init_pos[0]
    dy = desired_pos[1] - init_pos[1]
    
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        yaw = 0.0
    else:
        yaw = np.arctan2(dy, dx)
    
    euler = np.array([0.0, 0.0, yaw], dtype=np.float64)
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_euler2Quat(quat, euler, 'xyz')
    
    return quat

def quat_conj(q):
    """Conjugate of quaternion
    @param q quaternion
    @return conjugate
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_multiplication(q1, q2):
    """Hamilton product of 2 quaternions.
    @param q1 quaternion 
    @param q2 quaternion
    @return multiplication of quaternions
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
      w1*w2 - x1*x2 - y1*y2 - z1*z2,
      w1*x2 + x1*w2 + y1*z2 - z1*y2,
      w1*y2 - x1*z2 + y1*w2 + z1*x2,
      w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_to_yaw(q):
    """Get yaw from quaternion.
    @param q quaternion 
    @return yaw
    """
    w,x,y,z = q
    return math.atan2(2*(w*z + x*y),
                      1 - 2*(y*y + z*z))

def rad_to_max_pi(rad):
    """Make the given radian between -pi and pi
    @param rad
    @return radian between -pi and pi
    """
    return (rad + math.pi) % (2 * math.pi) - math.pi

def line_clear(p, q, tiles):
    """Check if the straight line from p to q is made of free tiles
    @param p first point
    @param q second point
    @return True if line is made of free tiles"""
    x0, y0 = map(int, map(round, p))
    x1, y1 = map(int, map(round, q))

    if x0 == x1:
        step = 1 if y1 > y0 else -1
        for y in range(y0, y1 + step, step):
            if tiles.get((x0, y), '#') != '.':
                return False
    elif y0 == y1:
        step = 1 if x1 > x0 else -1
        for x in range(x0, x1 + step, step):
            if tiles.get((x, y0), '#') != '.':
                return False
    else:
        return False
    return True

def relocate_corners(path, tiles):
    """Replace L shaped path for lesser number of turns if possible.
    @param path (x, y) points from planner
    @param tiles dungeon tiles
    @return updated path
    """
    new_path = [path[0]]

    for i in range(1, len(path) - 1):
        A = new_path[-1]
        B = path[i]
        C = path[i+1]

        Ax, Ay = map(int, map(round, A))
        Bx, By = map(int, map(round, B))
        Cx, Cy = map(int, map(round, C))

        if (Ax == Bx == Cx) or (Ay == By == Cy):
            new_path.append(B)
            continue

        corner1 = (Cx, Ay) 
        corner2 = (Ax, Cy)
        candidates = []
        for Dx, Dy in (corner1, corner2):
            if tiles.get((Dx, Dy), '#') == '.' \
               and line_clear((Ax, Ay), (Dx, Dy), tiles) \
               and line_clear((Dx, Dy), (Cx, Cy), tiles):
                dist = abs(Dx - Ax) + abs(Dy - Ay)
                candidates.append(((Dx, Dy), dist))

        if candidates:
            D, _ = min(candidates, key=lambda x: x[1])
            new_path.append((float(D[0]), float(D[1])))
        else:
            new_path.append(B)

    new_path.append(path[-1])
    return new_path

def remove_intermediate_point(path):
    """Remove any intermediate point in straight line
    @param path (x, y) path from planner
    @return simplified path
    """
    if len(path) <= 2:
        return path
    new = [path[0]]
    for A, B, C in zip(path, path[1:], path[2:]):
        if not ((A[0] == B[0] == C[0]) or (A[1] == B[1] == C[1])):
            new.append(B)
    new.append(path[-1])
    return new

def simplify_path(path, tiles):
    """Iteratively relocate L shaped corners until the path stops changing.
    @param path (x, y) points from the planner
    @param tiles dungeon tiles
    @return path with less turns
    """
    def step(path):
        p = remove_intermediate_point(path)
        return relocate_corners(p, tiles)

    prev = None
    curr = path
    while curr != prev:
        prev = curr
        curr = step(curr)
    return curr