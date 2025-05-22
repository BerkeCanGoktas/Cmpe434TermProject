import numpy as np
from python.utils import find_intersection

class Controller:
    def __init__(self, path, lookahead_distance, gain=1.0, throttle=1.0):
        """
        @param path waypoints list
        @param lookahead_distance look ahead distance
        @param gain steering gain
        @param throttle constant throttle
        """
        self.path = np.array(path)
        self.lookahead_distance = lookahead_distance
        self.base_throttle = throttle
        self.throttle = throttle
        self.gain = gain
        self.current_index = 0 
        
    def find_target_point(self, position):
        """Finds valid target points
        @param position robot position
        @return target point
        """
        while self.current_index < len(self.path) - 1:
            # print(f"Total wp: {len(self.path)} Current wp: {self.current_index}")
            dist_to_waypoint = np.linalg.norm(self.path[self.current_index] - position)
            next_dist = np.linalg.norm(self.path[self.current_index + 1] - position)

            if next_dist < dist_to_waypoint: 
                self.current_index += 1
                if self.current_index == len(self.path) - 1:
                    self.current_index = len(self.path) - 1
            else:
                break

        for i in range(self.current_index, len(self.path) - 1):
            A = self.path[i]
            B = self.path[i + 1]
            intersections = find_intersection(position, self.lookahead_distance, A, B)

            if intersections:
                return min(intersections, key=lambda pt: np.linalg.norm(pt - B))

        closest_index = np.argmin([
            np.linalg.norm(wp - position)
            for wp in self.path[self.current_index:]
        ])
        closest_point = self.path[self.current_index + closest_index]
        return closest_point
        
    def compute_steering(self, position, heading, target_point):
        """Computes the steering angle in radians.
        @param position current position of the robot
        @heading current heading of the robot
        @target_point desired look ahead point
        @return steering angle
        """
        dx = target_point[0] - position[0]
        dy = target_point[1] - position[1]

        if self.lookahead_distance == 0:
            curvature = 0
        else:
            curvature = (2 * np.sin(-heading) * dx + np.cos(-heading) * dy) / (self.lookahead_distance**2)
        
        steering_angle = np.arctan(curvature)
        return steering_angle
        
    def control_command(self, position, heading):
        """Returns final steering value
        @param position position of the robot
        @param heading heading of the robot
        @return steering value
        """
        target_point = self.find_target_point(position)
        if target_point is None:
            steering_angle = 0.0
        else:
            steering_angle = self.compute_steering(position, heading, target_point)
            self.throttle = self.base_throttle * max(0.2, 1.0 - 0.5 * abs(steering_angle))
        return self.gain * steering_angle

