"""
SimWorld iteration 1 - Humanoid navigation with LLM agent.

Converted from: ~/workspaces/SimWorld/iterations/iteration_1.ipynb

Usage:
    python simworld_iteration_1.py
    python simworld_iteration_1.py --model nvidia/Cosmos-Reason2-2B
    python simworld_iteration_1.py --steps 20
"""

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path

import httpx
import openai

# Add SimWorld to path
SIMWORLD_ROOT = Path(os.environ.get("SIMWORLD_ROOT", os.path.expanduser("~/workspaces/SimWorld")))
sys.path.append(str(SIMWORLD_ROOT))

from simworld.agent.humanoid import Humanoid
from simworld.communicator.communicator import Communicator
from simworld.communicator.unrealcv import UnrealCV
from simworld.config import Config
from simworld.map.map import Map
from simworld.utils.vector import Vector

OPENAI_BASE_URL = "https://embodied-ai-cosmos-reason.apps.octocp.alexnet.htx/v1"


class Environment:
    def __init__(self, communicator, config=Config()):
        self.communicator = communicator
        self.agent = None
        self.agent_name = None
        self.agent_spawned = False
        self.target = None
        self.config = config
        self.map = Map(config)
        self.map.initialize_map_from_file(
            roads_file=str(SIMWORLD_ROOT / "data/example_city/demo_city_1/roads.json")
        )

    def reset(self):
        agent_bp = "/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C"
        spawn_location = Vector(0, 0)
        spawn_forward = Vector(1, 0)

        if not self.agent_spawned:
            self.agent = Humanoid(
                communicator=self.communicator,
                position=spawn_location,
                direction=spawn_forward,
                config=self.config,
                map=self.map,
            )
            self.communicator.spawn_agent(self.agent, name=None, model_path=agent_bp, type="humanoid")
            self.communicator.humanoid_set_speed(self.agent.id, 200)
            self.agent_name = self.communicator.get_humanoid_name(self.agent.id)
            self.agent_spawned = True
        else:
            location_3d = [spawn_location.x, spawn_location.y, 600]
            orientation_3d = [0, math.degrees(math.atan2(spawn_forward.y, spawn_forward.x)), 0]
            self.communicator.unrealcv.set_location(location_3d, self.agent_name)
            self.communicator.unrealcv.set_orientation(orientation_3d, self.agent_name)
            spawn_yaw = math.degrees(math.atan2(spawn_forward.y, spawn_forward.x))
            self.agent.position = spawn_location
            self.agent.direction = spawn_yaw

        self.target = Vector(1700, -1700)
        time.sleep(5)

        return self._get_observation()

    def delete_agent(self):
        if self.agent_spawned and self.agent_name:
            self.communicator.unrealcv.destroy(self.agent_name)
            self.agent = None
            self.agent_name = None
            self.agent_spawned = False

    def step(self, action):
        action_cleaned = action.strip().strip('"').strip("'")
        action_lower = action_cleaned.lower().strip()
        success = False

        if action_lower.startswith("forward"):
            match = re.search(r"forward\s+(\d+\.?\d*)", action_lower)
            if match:
                duration = float(match.group(1))
                self.communicator.humanoid_step_forward(self.agent.id, duration, direction=0)
                success = True
            else:
                print(f"[Warning] Failed to parse forward action: '{action_cleaned}'")
        elif action_lower.startswith("rotate"):
            match = re.search(r"rotate\s+(\d+\.?\d*)\s+(left|right)", action_lower)
            if match:
                angle = float(match.group(1))
                direction = match.group(2)
                self.communicator.humanoid_rotate(self.agent.id, angle, direction)
                success = True
            else:
                print(f"[Warning] Failed to parse rotate action: '{action_cleaned}'")
        elif action_lower == "wait":
            time.sleep(1)
            success = True
        else:
            print(f"[Warning] Unknown action: '{action_cleaned}'")
            time.sleep(0.5)

        # Update agent state
        loc_3d = self.communicator.unrealcv.get_location(self.agent_name)
        position = Vector(loc_3d[0], loc_3d[1])
        orientation = self.communicator.unrealcv.get_orientation(self.agent_name)
        yaw = orientation[1]
        self.agent.position = position
        self.agent.direction = yaw

        observation = self._get_observation()
        reward = -position.distance(self.target)
        return observation, reward, success

    def _get_observation(self):
        loc_3d = self.communicator.unrealcv.get_location(self.agent_name)
        position = Vector(loc_3d[0], loc_3d[1])
        orientation = self.communicator.unrealcv.get_orientation(self.agent_name)
        yaw = orientation[1]
        direction = Vector(math.cos(math.radians(yaw)), math.sin(math.radians(yaw)))
        ego_view = self.communicator.get_camera_observation(self.agent.camera_id, "lit")
        return {"position": position, "direction": direction, "ego_view": ego_view}


class Agent:
    def __init__(self, model="nvidia/Cosmos-Reason2-8B"):
        # Create httpx client that skips SSL verification
        http_client = httpx.Client(verify=False)
        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=OPENAI_BASE_URL,
            http_client=http_client,
        )
        self.model = model
        self.system_prompt = (
            "You are an intelligent agent in a 3D simulation world.\n"
            "Your task is to navigate to the target position by choosing appropriate actions. "
            "Do not keep rotating in circles.\n\n"
            "You can only output ONE of the following actions:\n"
            '- "forward <duration>": Move forward for <duration> seconds (e.g., "forward 2")\n'
            '- "rotate <angle> <direction>": Rotate <angle> degrees in <direction> (left/right) '
            '(e.g., "rotate 45 left")\n'
            '- "wait": Do nothing for 1 second\n\n'
            "Output ONLY the action command, nothing else."
        )

    def action(self, obs, target):
        position = obs["position"]
        direction = obs["direction"]
        current_yaw = math.degrees(math.atan2(direction.y, direction.x))

        delta_x = target.x - position.x
        delta_y = target.y - position.y
        target_angle = math.degrees(math.atan2(delta_y, delta_x))

        angle_diff = target_angle - current_yaw
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360

        prompt = (
            f"Current position: {position}\n"
            f"Current direction: {direction} (direction vector, yaw: {current_yaw:.1f}°)\n"
            f"Target position: {target}\n"
            f"Distance to target: {position.distance(target):.1f}\n"
            f"Angle to target: {angle_diff:.1f}° (positive = turn left, negative = turn right)\n\n"
            f"Choose your next action to move closer to the target. The angle does not have to exactly be 0."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="SimWorld humanoid navigation")
    parser.add_argument("--model", default="nvidia/Cosmos-Reason2-8B", help="LLM model name")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to run")
    args = parser.parse_args()

    communicator = Communicator(UnrealCV())

    agent = Agent(model=args.model)

    env = Environment(communicator=communicator)

    obs = env.reset()
    print(f"Initial position: {obs['position']}")
    print(f"Initial direction: {obs['direction']}")
    print(f"Target: {env.target}")

    for step in range(args.steps):
        action = agent.action(obs, env.target)
        if action is None:
            print("Quit requested.")
            break

        print(f"\n[Step {step + 1}] Action: {action}")
        obs, reward, success = env.step(action)
        position = obs["position"]
        direction = obs["direction"]
        current_yaw = math.degrees(math.atan2(direction.y, direction.x))
        print(f"  Position: {position}, Direction: {direction} (yaw: {current_yaw:.1f}°)")
        print(f"  Reward: {reward:.2f}, Distance to target: {position.distance(env.target):.1f}")

    env.delete_agent()
    print("Done.")


if __name__ == "__main__":
    main()
