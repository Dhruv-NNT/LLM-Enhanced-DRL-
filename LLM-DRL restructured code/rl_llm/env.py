"""Environment and scenario classes derived from RL - LLM (Complete Code Pipeline).ipynb."""
# Math helps us wrap angles and perform trigonometric updates.
import math
# Path lets us work with filesystem paths in an easy way.
from pathlib import Path
# Optional is used for type hints that may be missing values.
from typing import Optional

# Gymnasium provides the RL environment interface we implement.
import gymnasium as gym
# Spaces helps define the observation and action spaces.
from gymnasium import spaces
# NumPy powers vector math for positions and headings.
import numpy as np
# Pandas loads the scenario feature file into a DataFrame.
import pandas as pd
# Matplotlib lets us plot the sector and trajectories for debugging.
import matplotlib.pyplot as plt
# Shapely gives us geometry helpers for path projections.
from shapely.geometry import LineString, Point

# We load the path to the feature file from the config module.
from configs import FEATUREFILE_PATH
# Utility helpers load scenarios, plot sectors, and project onto paths.
from .utils import get_conflict_scenario, plot_sector, project_path

# We read the full feature file once so every class can use it.
featurefile = pd.read_csv(FEATUREFILE_PATH)

class Ownship:
    def __init__(self, start_x, start_y, heading, speed, offset, original_path, destination):
        # We remember the starting x coordinate.
        self.start_x = start_x
        # We remember the starting y coordinate.
        self.start_y = start_y
        # We track the current x position, starting at the origin point.
        self.x = start_x
        # We track the current y position, starting at the origin point.
        self.y = start_y
        # We store how many time steps to wait before moving.
        self.offset = offset
        # We keep the current heading in radians.
        self.heading = heading
        # We remember the movement speed per step.
        self.speed = speed
        # We keep the same offset again for clarity.
        self.offset = offset
        # We remember where the plane should end up.
        self.destination = destination
        # We collect the flown path for later plotting.
        self.pathlist = []
        # We store the original planned path for comparison.
        self.original_path = original_path
        # We track the current position as a list for easy mutation.
        self.position = [self.x, self.y]

    def step(self, n_step, new_heading):
        # If we have not reached the offset yet we stay at the start point.
        if n_step <= self.offset:
            self.x = self.start_x
            self.y = self.start_y
        else:    
            # We update the heading once the aircraft begins moving.
            self.heading = new_heading
            # We move forward along the new heading, clamped to the map edges.
            self.x = np.clip(self.x + self.speed *np.cos(new_heading), 0, 100)
            self.y = np.clip(self.y + self.speed *np.sin(new_heading), 0, 100)
        # We return the new position after this step.
        return self.x, self.y
    
class Intruder:
    def __init__(self, start_x, start_y, offset, heading, speed, destination):
        # We remember the starting x coordinate.
        self.start_x = start_x
        # We remember the starting y coordinate (typo kept for compatibility).
        self.stat_y = start_y
        # We track the current x position, starting at the origin point.
        self.x = start_x
        # We track the current y position, starting at the origin point.
        self.y = start_y
        # We store the wait time before the intruder starts moving.
        self.offset = offset
        # We store the current heading in radians.
        self.heading = heading
        # We keep the movement speed per step.
        self.speed = speed
        # We remember the destination point.
        self.destination = destination
        # We track the current position for plotting.
        self.position = [self.x, self.y]

    def step(self, n_step, new_heading):
        # If we have not reached the offset yet we hold the starting position.
        if n_step <= self.offset:
            self.x = self.start_x
            self.y = self.start_y
        else:
            # Otherwise we update the heading and move forward.
            self.heading = new_heading
            self.x = np.clip(self.x + self.speed *np.cos(new_heading), 0, 100)
            self.y = np.clip(self.y + self.speed *np.sin(new_heading), 0, 100)
        # We return the updated position.
        return self.x, self.y


class Agent():
    """
    The Agent class encapsulates the state and parameters for two aircraft in a conflict scenario: the “ownship” (o_) 
    and the “intruder” (i_). Its __init__ method calls a helper to load all scenario data (paths, headings, start/destination
    points, scaled offsets, etc.) for one scenario from our DataFrame, then organizes that into easy-to-use attributes. It 
    also computes fixed speeds and the number of time-steps to reach each aircraft’s first conflict point (“offset”), and 
    prepares a LineString for the ATCO-approved path.

        # 1) Pull all relevant scenario arrays/values for ownship (o_*) and intruder (i_*).
        #    get_conflict_scenario returns:
        #    - o_resolved_path: list of [x, y] for ATCO-approved ownship route
        #    - o_unres_path:   list of [x, y] for original ownship route before conflict resolution
        #    - o_heading:      initial ownship heading (radians)
        #    - o_startposition:ownship start [x, y]
        #    - o_offset_dist_scaled: distance ownship travels before conflict point (scaled)
        #    - o_destination:  ownship destination [x, y]
        #    - i_path:         intruder path coordinates list
        #    - i_headinglist:  list of heading angles for intruder at each step
        #    - i_startposition:intruder start [x, y]
        #    - i_offset_dist_scaled: intruder’s scaled offset distance
        #    - i_destination:  intruder destination [x, y]
        #    - fileinfo:       metadata about this scenario (e.g., filename, index)
    """
    def __init__(self):
        # We pull one random conflict scenario out of the dataset.
        o_resolved_path, o_unres_path, o_heading, o_startposition, o_offset_dist_scaled, o_destination, \
        i_path, i_headinglist, i_startposition, i_offset_dist_scaled, i_destination, fileinfo = get_conflict_scenario(featurefile)
        # 2) Store ownship’s “resolved” (ATCO-approved) vs. “unresolved” paths
        # We keep the resolved path so the agent knows the safe route.
        self.o_resolved_path = o_resolved_path
        # We also keep the original unresolved path for reference.
        self.o_unresolved_path = o_unres_path
        # 3) Initial and current heading for ownship
        def _wrap_pi(a: float) -> float:
            # We wrap any angle into the [-pi, pi) range.
            return ((float(a) + math.pi) % (2.0 * math.pi)) - math.pi

        # We store the initial heading in wrapped form.
        self.o_initial_heading = _wrap_pi(o_heading)
        # We also track the mutable heading during flight.
        self.o_heading = self.o_initial_heading

        # 4) Start position (tuple) vs. mutable position list
        # We remember the start as an immutable tuple.
        self.o_start = tuple(o_startposition)
        # We also keep a mutable copy we can update as the plane moves.
        self.o_position = o_startposition
        # 5) Distance until conflict (“offset”), scaled for ownship
        # This distance tells us when the conflict begins for ownship.
        self.o_offset_dist = o_offset_dist_scaled
        # 6) Final waypoint for ownship
        # We store the final destination point for ownship.
        self.o_destination = o_destination
            # --- Ensure the resolved (ATCO) path runs from START → DESTINATION ---
        def _ensure_forward_path(path, start, dest):
            # If the path is saved backwards (dest→start), reverse it.
            # We compare which ordering is closer to (start, dest) at the ends.
            p0 = np.array(path[0], dtype=float)
            pN = np.array(path[-1], dtype=float)
            start = np.array(start, dtype=float)
            dest  = np.array(dest, dtype=float)
            keep_cost = np.linalg.norm(p0 - start) + np.linalg.norm(pN - dest)
            flip_cost = np.linalg.norm(pN - start) + np.linalg.norm(p0 - dest)
            return path if keep_cost <= flip_cost else list(reversed(path))

        if len(o_resolved_path) >= 2:
            # We make sure the stored resolved path points in the right direction.
            self.o_resolved_path = _ensure_forward_path(o_resolved_path, self.o_start, self.o_destination)
        else:
            # Degenerate path (just in case) – keep as provided
            self.o_resolved_path = o_resolved_path

        # 7) Record actual flown trajectory
        # We start the flown path list with the starting position.
        self.o_newpathlist = [self.o_position]

        # 8) Build LineString **after** possibly reversing the path
        # We create a shapely line so we can project the plane onto the path.
        self.ATCO_path_linestring = LineString(self.o_resolved_path)

        # 9) Intruder: headings & positions
        # We store the intruder headings sequence.
        self.i_headinglist = i_headinglist
        # We save the intruder start position as an immutable tuple.
        self.i_start = tuple(i_startposition)
        # We also keep a mutable intruder position.
        self.i_position = i_startposition

        # 10) Intruder path/offset/destination
        # We keep the intruder path for reference.
        self.i_path = i_path
        # We store the distance until the intruder reaches the conflict.
        self.i_offset_dist = i_offset_dist_scaled
        # We record the intruder destination point.
        self.i_destination = i_destination

        # 11) Fixed speeds (units/step)
        # Both aircraft move at a constant speed of two units per step.
        self.o_speed = 2
        self.i_speed = 2

        # 12) Steps to each offset
        # We convert the conflict distance into time steps for ownship.
        self.o_offset = int(np.ceil(self.o_offset_dist / self.o_speed))
        # We do the same conversion for the intruder.
        self.i_offset = int(np.ceil(self.i_offset_dist / self.i_speed))

        # 13) Count of applied actions
        # We reset the action counter for logging decisions.
        self.n_actions = 0

        # 14) Scenario metadata
        # We store the metadata about the loaded scenario.
        self.fileinfo = fileinfo

    def step(self, n_step, new_heading):
        # Prepare placeholders for new x,y positions
        xo, yo, xi, yi = 0, 0, 0, 0

        # --- Move the ownship ---
        # OLD-PPO-compatible behavior:
        # - Do NOT change self.o_heading until the ownship actually starts moving
        # - Do NOT wrap the angle to (-pi, pi]; keep it as-is
        if n_step <= self.i_offset:
            # Not moving yet: keep position and keep the previous heading unchanged
            self.o_position = self.o_start
        else:
            # Update heading only now (when we start moving), unwrapped
            self.o_heading = new_heading

            # Move forward by o_speed along that heading (clipped to bounds)
            xo = np.clip(self.o_position[0] + self.o_speed*np.cos(new_heading), 15, 100)
            yo = np.clip(self.o_position[1] + self.o_speed*np.sin(new_heading), 15, 100)
            self.o_position = [xo, yo]

        # Save the actual current ownship position
        self.o_newpathlist.append(list(self.o_position))

        # --- Intruder update (unchanged) ---
        if n_step <= self.o_offset:
            self.i_position = self.i_start
        elif len(self.i_path) + self.o_offset > n_step > self.o_offset:
            xi = np.clip(self.i_position[0] + self.i_speed*np.cos(self.i_headinglist[n_step - self.o_offset]), 15, 100)
            yi = np.clip(self.i_position[1] + self.i_speed*np.sin(self.i_headinglist[n_step - self.o_offset]), 15, 100)
            self.i_position = [xi, yi]
        else:
            self.i_position = self.i_destination

        return self.o_position, self.i_position

        
    def path_follow_heading(self, lookahead: float = 1.0) -> float:
        """
        Aim a little ahead on the ATCO path (pure-pursuit). Returns heading in radians.
        """
        # We turn the current ownship location into a shapely point.
        cur = Point(self.o_position[0], self.o_position[1])
        # We find how far along the ATCO path this point sits.
        s = self.ATCO_path_linestring.project(cur)
        # We look a bit ahead along the path to smooth the heading command.
        s_next = min(self.ATCO_path_linestring.length, s + lookahead)
        # We pick the target point at that lookahead distance.
        tgt = self.ATCO_path_linestring.interpolate(s_next)
        # We compute the x offset between the target and the current plane position.
        dx = tgt.x - self.o_position[0]
        # We compute the y offset between the target and the current plane position.
        dy = tgt.y - self.o_position[1]
        # We convert the offsets into a desired heading angle.
        return math.atan2(dy, dx)

class StructuredEnv(gym.Env):
    """
    This constructor sets up our custom Gym environment. It takes in a list of reward-weight parameters, 
    creates a fresh Agent. It initializes step and reward counters, then defines how many actions are possible 
    and what shape/value-ranges our observations will have. This tells any RL algorithm how to talk to our environment.
    """
    # Minimum safe separation distance between aircraft in grid units.
    SAFE_R = 5 # Minimum safe separation distance between aircraft
    # Hard cap on how many steps the episode can run.
    MAX_STEP = 60 # Maximum number of time-steps per episode

    def __init__(self, Reward_Params= [-1, -1, -1, -1, -1, 1], start_llm_at_step: int = 15):
        # 1) Store the list of six reward-component weights (a1…a6)
        # We store the reward weights so the reward function can use them later.
        self.Reward_Params = Reward_Params
        # 2) Create a new conflict-scenario agent (ownship + intruder)
        # We create a brand new agent with fresh aircraft positions.
        self.agent = Agent()
        # 3) Initialize the reward from the last step, the current step counter and the accumulated total reward
        # We reset the immediate reward value.
        self.reward = 0
        # We reset the step counter back to zero.
        self.n_step = 0
        # We reset the total episode reward tracker.
        self.total_reward = 0
        # Actions are 13 discrete turn bins from -30° to +30°.
        self.action_space = spaces.Discrete(13)
        # We compute one observation to learn the vector length.
        first_obs = self.observation_func()
        n = first_obs.size

        # We start with very wide observation bounds.
        low = np.full(n, -np.inf, dtype=np.float64)
        high = np.full(n,  np.inf, dtype=np.float64)

        # 0..5: positions/destination coords (you clip positions inside [15,100], keep generous but finite)
        low[0:6]  = -100.0
        high[0:6] =  120.0

        # 6: separation distance >= 0
        low[6]  = 0.0

        # 7..8: distances to destinations >= 0
        low[7:9] = 0.0

        # 9: cross-track distance to unresolved path (distance) >= 0
        low[9] = 0.0

        # 10: distance to ATCO path (distance) >= 0
        low[10] = 0.0

        # 11: own heading (radians)
        low[11]  = -math.pi
        high[11] =  math.pi

        # 12: ATCO path heading (degrees)
        low[12]  = -180.0
        high[12] =  180.0

        # 13: heading error to path (degrees)
        low[13]  = -180.0
        high[13] =  180.0

        # 14: signed cross-track to path (can be ± large) -> leave as ±inf (already set)

        # 15..16: along-track s and remaining path length >= 0
        low[15:17] = 0.0

        # We register the observation space so Gym knows valid ranges.
        self.observation_space = spaces.Box(low=low, high=high, shape=(n,), dtype=np.float64)
        # We store when the controller should start asking the LLM.
        self.start_llm_at_step = start_llm_at_step  # call LLM only once ≥ this controller step

    def reset(self, seed=None, options=None):
        """
        The reset method is called at the start of each episode to bring the environment back to its initial state. 
        It optionally seeds randomness, creates a fresh Agent (ownship + intruder), zeroes out all counters 
        and rewards, computes the very first observation, and returns that observation along with an empty info dictionary.
        """
        super().reset(seed=seed) # Call parent reset if you pass seed/options
        # Create a new Agent instance for each reset
        self.agent = Agent() 
        # Initialize counters and rewards, Clear out the last step reward 
        self.reward = 0 
        self.n_step = 0
        self.total_reward = 0
        # Compute the initial observation vector from the fresh agent state
        observation = self.observation_func()
        # Gym’s `reset` requires an `info` dict (we have no extra data to return)
        info = {} 
        # Return the tuple (observation, info)
        # This is the standard return format for Gym environments
        # The observation is the initial state of the environment, and info is an empty dictionary
        return observation, info 

    def step(self, action):
        """
        Given an action (one of 13 discrete turn adjustments), 
        it advances the global time-step, converts that action into a small heading change for the ownship, 
        updates both aircraft positions via the Agent.step call, checks if the episode should end 
        (because both planes reached their goals, came too close, moved away, or exceeded the maximum steps), 
        computes the reward based on all those factors, and finally returns the new observation, the step reward, 
        and flags indicating whether the episode terminated or was truncated.

        We chose 0.175/2 because:
        0.175 radians is about 10° (since 1° ≈ 0.01745 rad).
        By dividing by 2, you get ≈0.0875 rad, which is about 5°.
        That way each “1-step” in your discrete action corresponds to a 5° turn. 
        Multiplying that base step by 0–6 (or –6 to –1) then gives you turns of 0°, ±5°, ±10°, … up to ±30° per time-step.
        """
        # 1) Advance the step counter
        self.n_step += 1
        
        # 2) Prepare termination/truncation flags
        terminated = False
        truncated = False
        # --- 3) Choose the base heading depending on the step ---
        # Steps 0–4: autopilot hugs the ATCO path.
        # Step ≥ 5: autopilot OFF → base is current heading; only LLM deltas change it.
        if self.n_step < self.start_llm_at_step:
            # Early in the episode we follow the planned ATCO heading.
            base_heading = self.agent.path_follow_heading(lookahead=1.0)
        else:
            # Later on we use the current heading and rely on actions to adjust.
            base_heading = self.agent.o_heading  # keep flying what we have unless LLM adds a delta

        # --- Discrete delta from action (0.175/2 ≈ 5° per bin) ---
        if action < 7:
            # Actions 0-6 add positive multiples of 5 degrees.
            angle = action * (0.175/2)          # +0,+5,...,+30 deg
        else:
            # Actions 7-12 subtract 5 degree steps.
            angle = (action - 6) * -(0.175/2)   # -5,...,-30 deg

        # Final heading this step
        # We combine the base heading with the discrete change.
        ownship_heading = base_heading + angle


        # 4) Move both aircraft one step using the composed heading
        # We update both the ownship and intruder states.
        self.agent.step(self.n_step, ownship_heading)

        # 5) Measure distances for termination checks:
        #    - First extract the points 
        #    - Ownship to its destination
        #    - Intruder to its destination
        #    - Separation between the two aircraft
        # We grab the current ownship position.
        p1 = self.agent.o_position
        # o_destination = self.agent.o_destination
        # We cast the ownship destination to a NumPy array for math.
        o_destination = np.array(self.agent.o_destination, dtype=float)
        # We grab the current intruder position.
        p2 = self.agent.i_position
        # We store the intruder destination.
        i_destination = self.agent.i_destination
        # We measure how far ownship is from its goal.
        dist1_to_dest = np.linalg.norm(np.array(p1) - np.array(o_destination))
        # We measure how far intruder is from its own goal.
        dist2_to_dest = np.linalg.norm(np.array(p2) - np.array(i_destination))
        # We measure the separation distance between both aircraft.
        sep_dist = np.linalg.norm(np.array(p1) - np.array(p2))

        # 6) Terminate if both reached within 5 units of their goals
        if (dist1_to_dest < 5) and (dist2_to_dest < 5):
            # Both aircraft are safely near their goals, so we end the episode.
            terminated = True
        
        # 7) Terminate if they come closer than SAFE_R (collision risk)
        if (sep_dist < self.SAFE_R):
            # If they are too close we stop immediately due to collision risk.
            terminated = True

        # 8) Terminate if ownship is moving away from its goal by more than a small threshold
        threshold = 0.1
        if len(self.agent.o_newpathlist) > 2: # Ensure there's a previous point to compare
            previous_location = np.array(self.agent.o_newpathlist[-2])
            o_current_location = np.array(self.agent.o_position)
            prev_distance = np.linalg.norm(o_destination - previous_location)
            current_distance = np.linalg.norm(o_destination - o_current_location)
            if prev_distance - current_distance < -threshold:
                # If the ownship is drifting away we also stop the run.
                terminated = True # Punish for moving away

        # 9) Truncate if we exceed MAX_STEP
        if self.n_step >= self.MAX_STEP:
            # Episodes longer than MAX_STEP are truncated.
            truncated = True
        
        # 10) Compute the reward, passing a combined “done” flag
        combined_done_for_reward_func = terminated or truncated
        # We calculate the reward with knowledge of whether the episode ended.
        self.reward = self.reward_func(combined_done_for_reward_func)
        # We add this step's reward to the running total.
        self.total_reward += self.reward

        # 11) Build next observation and empty info dict
        observation = self.observation_func()
        # We return an empty info dictionary as required by Gym.
        info = {}
        # 12) Return the Gym-standard 5-tuple
        return observation, self.reward, terminated, truncated, info # Return 5 values

            
    def reward_func(self, done):
        """ 
        This method computes the scalar reward at each step by combining several components—step penalty, 
        trajectory deviation, loss-of-separation penalty/bonus, failure-to-reach penalty, and success bonus—each 
        weighted by the six parameters a1…a6. If the episode has ended (done=True), additional terms (e.g., whether
        we collided or reached the goal) are applied. The final reward is the weighted sum plus any extra “moving-back” 
        penalty.

        """
        # 1) Unpack the six weights from the environment’s configuration        
        a1 = self.Reward_Params[0]
        a2 = self.Reward_Params[1]
        a3 = self.Reward_Params[2]
        a4 = self.Reward_Params[3]
        a5 = self.Reward_Params[4]
        a6 = self.Reward_Params[5]
        
        # 2) Base costs or rewards (initialized here)
        Reward_Step = 0.01 # small per-step penalty to encourage faster resolution
        Moving_back_reward = 0 # penalty if ownship moves away from its goal
        CTDreward = 0 # placeholder for distance-from-original-path reward (unused)
        reward_traj_dev = 0 # trajectory deviation reward


        # 3) Compute deviation from the ATCO-approved path:
        #    • Create a Point for ownship’s current pos
        #    • Compute its distance to the approved LineString (in same units)
        #    • Scale by 20/1000 to form the deviation reward
        # We measure how far the ownship is from the destination.
        o_destination = np.array(self.agent.o_destination)       
        # We create a point out of the current ownship position.
        o_current_location = Point(self.agent.o_position)
        # We grab the resolved ATCO path line.
        ATCO_path_linestring = self.agent.ATCO_path_linestring
        # We compute the shortest distance from ownship to the ATCO path.
        dist = o_current_location.distance(ATCO_path_linestring)
        # We scale the distance to form a small penalty.
        reward_traj_dev =  20 * (dist/200)
        # 4) Combine any CTDreward (currently zero) with the trajectory deviation
        Reward_Deviation = CTDreward + reward_traj_dev # This is negative, so it penalizes deviation
        
        # 5) Initialize other components
        Reward_LOS = 0
        Reward_Maneuvering = 0
        Reward_Not_Reached = 0
        Reward_Reached =0

        # 6) Distance between ownship and intruder (for LOS/collision)
        dis1 = np.linalg.norm(np.array(self.agent.o_position) - np.array(self.agent.i_position))
        
        Reward_LOS = 0
        threshold = 0.1
        # 7) Only apply certain rewards/penalties once the episode is done
        if done:
            # 7a) Collision or near-miss bonus/penalty: if we ended due to separation breach
            if (dis1 < self.SAFE_R) :
                Reward_LOS = 10 # This is a negative reward, so it penalizes loss of separation

            # 7b) If we reached the maximum number of steps without reaching the destination
            if self.n_step == self.MAX_STEP:
                # Penalize based on how far ownship still is from its destination
                current_location = self.agent.o_position
                dist_o_d = np.linalg.norm(np.array(o_destination) - np.array(current_location))
                Reward_Not_Reached = (dist_o_d/10) # Penalize for not reaching destination

            # 7c) Penalty if the ownship just moved away from its destination in final step (done in the else part)
            if len(self.agent.o_newpathlist) <=2:
                pass
            else:
                # Grab the second-to-last position we recorded for ownship
                previous_location = np.array(self.agent.o_newpathlist[-2])
                # Grab the current position of ownship
                o_current_location = np.array(self.agent.o_position)
                # Compute distance from that previous point to the destination
                prev_distance = np.linalg.norm(o_destination - previous_location)
                # Compute distance from the final point to the destination
                current_distance = np.linalg.norm(o_destination - o_current_location)
                # If current_distance > prev_distance by more than `threshold`,
                # it means ownship moved away. We then set a penalty:
                if prev_distance - current_distance < -threshold:
                    Moving_back_reward =  -20
               

            #reaching destination
            p1 = self.agent.o_position
            o_destination = self.agent.o_destination
            p2 = self.agent.i_position
            
            i_destination = self.agent.i_destination
            # 7d) Success bonus or failure penalty upon reaching ownship destination (else condition)
            dist1 = np.linalg.norm(np.array(p1) - np.array(o_destination))
            # dist2 = np.linalg.norm(np.array(p2) - np.array(i_destination))
            if (dist1 < 5): # reaching destination
                Reward_Reached = 10
            else:
                Reward_Reached = - dist1/10 # penalize for not reaching destination (proportional to remaining distance)

        # We mix all parts using the provided weights.
        reward = a1 * Reward_Step + a2 *Reward_Deviation + a3 * Reward_LOS + \
                    a5 * Reward_Not_Reached + a6 * Reward_Reached + Moving_back_reward 
        # We return the final number back to the caller.
        return reward        


    def observation_func(self):
        """
        This method gathers all the state information we want the agent to “see” at each step,
        packs it into a flat NumPy array, and returns it. It includes the ownship’s and intruder’s positions,
        the ownship’s destination, pairwise distances (between aircraft and to each destination), how far the ownship
        has strayed from its original and ATCO-approved paths, and the ownship’s current heading.

        The observation vector contains, in order:
            Ownship’s X and Y position
            Intruder’s X and Y position
            Ownship’s destination X and Y
            Distance between ownship and intruder
            Distance from ownship to its destination
            Distance from intruder to its destination
            Distance from ownship to its original (unresolved) path
            Distance from ownship to the ATCO-approved path
            Ownship’s current heading angle
        """
        observation  = []

        # Ownship X coordinate.
        observation.append(self.agent.o_position[0])
        # Ownship Y coordinate.
        observation.append(self.agent.o_position[1])
        # Intruder X coordinate.
        observation.append(self.agent.i_position[0])
        # Intruder Y coordinate.
        observation.append(self.agent.i_position[1])
        # Ownship destination X coordinate.
        observation.append(self.agent.o_destination[0])
        # Ownship destination Y coordinate.
        observation.append(self.agent.o_destination[1])
        # observation.append(self.agent.i_destination[0])
        # observation.append(self.agent.i_destination[1])
    

        # Distance between ownship and intruder.
        dist1 = np.linalg.norm(np.array(self.agent.o_position) - np.array(self.agent.i_position)) 
        # Distance from ownship to its destination.
        dist_o_d = np.linalg.norm(np.array(self.agent.o_position) - np.array(self.agent.o_destination))
        # Distance from intruder to its destination.
        dist_i_d = np.linalg.norm(np.array(self.agent.i_position) - np.array(self.agent.i_destination))

        # Distance between ownship and intruder
        observation.append(dist1)  
        # Distance from ownship to its destination
        observation.append(dist_o_d)
        # Distance from intruder to its destination
        observation.append(dist_i_d)
        # individual coordinates

        # We build a point for the ownship location to measure path distances.
        current_location = Point(self.agent.o_position)
        # How far ownship is from its ORIGINAL unresolved path
        CTD_distance  = current_location.distance(LineString(self.agent.o_unresolved_path))
        # How far ownship is from the ATCO-approved path
        dist_from_ATCO_path = current_location.distance(self.agent.ATCO_path_linestring)

        # Distance to unresolved path.
        observation.append(CTD_distance) #
        # Distance to resolved ATCO path.
        observation.append(dist_from_ATCO_path)

        # Ownship’s current heading angle (radians)
        observation.append(self.agent.o_heading)  

        # New LLM observation:
        # --- NEW ATCO-path features ---
                # --- NEW ATCO-path features ---
        # We reuse shapely points to compute new path features.
        cur_pt = Point(self.agent.o_position[0], self.agent.o_position[1])
        line = self.agent.ATCO_path_linestring

        # Along-track distance s and foot point on the path
        s = line.project(cur_pt)
        foot = line.interpolate(s)

        # Unit tangent at s (finite-difference fallback)
        eps = max(1e-3, min(1.0, 0.001 * line.length))
        p_fwd = line.interpolate(min(line.length, s + eps))
        tx, ty = (p_fwd.x - foot.x, p_fwd.y - foot.y)
        if abs(tx) < 1e-9 and abs(ty) < 1e-9:  # fallback if degenerate
            h = self.agent.path_follow_heading(lookahead=1.0)
            tx, ty = math.cos(h), math.sin(h)

        # Signed cross-track: sign by 2D cross product (tangent × foot->own)
        dx, dy = (cur_pt.x - foot.x, cur_pt.y - foot.y)
        cross = tx * dy - ty * dx
        signed_xtrk = math.copysign(math.hypot(dx, dy), cross)

        # Convert tangent to degrees
        path_heading_deg = math.degrees(math.atan2(ty, tx))
        own_heading_deg  = math.degrees(self.agent.o_heading)
        # heading_error_to_path_deg = (path - own) wrapped to (-180, 180]
        heading_error_to_path_deg = ((path_heading_deg - own_heading_deg + 180.0) % 360.0) - 180.0

        # We add path heading, heading error, and signed cross-track in the expected order.
        observation.extend([
            path_heading_deg,           # index 12
            heading_error_to_path_deg,  # index 13
            signed_xtrk,                # index 14
        ])

        # (Optional) You can still add these extras; ObsSnapshot just ignores them:
        along_track_s = float(s)
        remaining_path_len = float(line.length - s)
        # We append how far along the path we are and how much path remains.
        observation.extend([along_track_s, remaining_path_len])  # indices 15,16 (optional)
   
        # Convert to a NumPy array and return
        observation = np.array(observation)

        # NEW LLM observation
        
        return observation

   
    def render(self, show: bool = False, folder: Optional[str] = None):
        """
        The render method draws a snapshot of the current episode on top of our sector plot, marking both aircraft’s 
        start, current, and destination points, the ATCO-approved path, and safety circles. It can either display the 
        figure interactively or save it to disk. The close method is a no-op placeholder for any future cleanup.
        """
        # We decide where to save images; default folder is under Images.
        folder_path = Path(folder) if folder is not None else project_path('Images')
        # 1) Draw the static sector background
        figure = plot_sector() # Call our helper to redraw boundaries/waypoints
        # We fetch the axes so we can add points and paths.
        ax = figure.gca()  # Grab the current Axes for plotting overlays
        # ax.scatter(50,50, s = 200)
        
        # 2)Plot ownship’s current position (black dot), Plot ownship’s start (black triangle), Plot ownship’s destination (black star)
        # Black dot marks current ownship position.
        ax.scatter(self.agent.o_position[0], self.agent.o_position[1], c = 'black', s = 40)
        # Black triangle marks ownship start.
        ax.scatter(self.agent.o_start[0], self.agent.o_start[1] , c = 'black', s = 50, marker = '^')
        # Black star marks ownship destination.
        ax.scatter(self.agent.o_destination[0], self.agent.o_destination[1], c = 'black', s = 20, marker= '*')

        # 3) Plot intruder’s current position (maroon dot), Plot intruder’s start (maroon triangle), Plot intruder’s destination (maroon star)
        # Maroon dot marks intruder current position.
        ax.scatter(self.agent.i_position[0], self.agent.i_position[1], c = 'maroon', s = 40)
        # Maroon triangle marks intruder start.
        ax.scatter(self.agent.i_start[0], self.agent.i_start[1] , c = 'maroon', s = 50, marker = '^')
        # Maroon star marks intruder destination.
        ax.scatter(self.agent.i_destination[0], self.agent.i_destination[1], c = 'maroon', s = 20, marker= '*')
        x, y = [],[]
        # We collect the unresolved path points in case we want to plot them later.
        for i in range(len(self.agent.o_unresolved_path)):
            x.append(self.agent.o_unresolved_path[i][0])    
            y.append(self.agent.o_unresolved_path[i][1])
            
        # ax.plot(x,y, linewidth = 1, color = 'grey')
        #add circle to the destination
        
        # 4) Draw the ATCO-approved path as a dashed maroon line
        # The ATCO path is drawn as a dashed maroon line.
        x_, y_ = self.agent.ATCO_path_linestring.xy
        ax.plot(x_, y_, color = 'maroon',linestyle = '--', alpha = 1)

        # 5) Add translucent safety circles around each destination
        circle1 = plt.Circle((self.agent.o_destination[0], self.agent.o_destination[1]), 5, color = 'black',\
                              alpha = 0.2)
        circle2 = plt.Circle((self.agent.i_destination[0], self.agent.i_destination[1]), 5, color = 'maroon',\
                              alpha = 0.5)
        # We add faint circles to show acceptable arrival area.
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        # 6) Annotate title with step number, recent reward, totals, headings, and offsets
        # Title summarises current timestep, reward stats, headings, and offsets.
        ax.set_title("Step {} - Reward {:.3f} - Total Reward {:.3f} - headings{}  - offsets(O:I){}".format(self.n_step,self.reward,\
                                                                            self.total_reward,\
                                                                                ((self.agent.o_initial_heading) ,\
                                                                                     np.round(math.degrees(self.agent.o_heading),2)),\
                                                                                        (self.agent.o_offset, self.agent.i_offset)))

        # ax.axis('scaled')
        
        # 7) Display the plot or save it to a file
        if show:
            # When asked we pop the figure on screen.
            plt.show()
        else:
            # Otherwise we ensure the folder exists and save to disk.
            folder_path.mkdir(parents=True, exist_ok=True)
            output_path = folder_path / f"image_{self.n_step:03d}.png"
            plt.savefig(output_path)
            print(str(output_path))

        return figure

    def close(self):
        # Placeholder for any cleanup (e.g., closing files or windows). Currently does nothing.
        pass
