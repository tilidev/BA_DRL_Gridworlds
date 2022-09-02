import time
import gym
import gym_minigrid
from gym_minigrid.envs import RiskyPathEnv
from gym_minigrid.envs.risky import DEFAULT_REWARDS
import random
import numpy as np
from stable_baselines3.dqn import DQN
from stable_baselines3.a2c import A2C
import json

from gym_minigrid.minigrid import Lava, Wall

def model_env_from_path(agent_path: str, no_slip: bool = False, no_rebound: bool = True):
    """Extract model, environment, observation type and tile render size
    from information in the model's save-path.
    """    
    # Extract model from path (a2c or dqn?)
    if "/dqn/" in agent_path:
        model_class = DQN
    elif "/a2c/" in agent_path:
        model_class = A2C

    model = model_class.load(agent_path)

    # Create environment given information in function input
    path_keys = agent_path.split("saved_models/")[1].split("/")
    env_name = path_keys[0]

    render_size = 8
    rgb = False
    if "pixel_obs_" in agent_path:
        render_size = int(path_keys[1].split("_")[-1])
        rgb = True

    env_info = ""
    with open('env_config.json', 'r') as f:
        env_kwargs = json.load(f)[env_name]
    if 'goal_rnd' in env_kwargs:
        env_kwargs.pop('goal_rnd')
    if no_slip and env_kwargs['slip_proba'] != 0:
        env_kwargs.pop('slip_proba')
        env_info += "slipping probability removed; "
    if no_rebound and env_kwargs['wall_rebound']:
        env_kwargs.pop('wall_rebound')
        env_info += "wall rebound deactivated"
    if len(env_info) != 0:
        print("\33[32mINFO:\33[0m", env_info)

    env = gym.make(
        "MiniGrid-RiskyPath-v0",
        **env_kwargs
    )

    return model, env, rgb, render_size

class RiskyPathSolver:
    def __init__(self, env: RiskyPathEnv, discount=0.99):
        """Utilities for computing value functions and optimal policies
        given certain environment configurations. E.g. slip
        probabilities. Currently does not work
        with wall rebound, time penalties (TODO) and spiky tiles.

        Args:
            env (RiskyPathEnv): RiskyPath environment
            discount (float): Reward discounting factor
        """

        assert env.reward_spec == DEFAULT_REWARDS, \
            "Certain reward models have not yet been implemented for the solver."
        # TODO currently: no wall_rebound, no time penalty

        self.gamma = discount
        self.slip_proba = env.slip_proba
        self.rnd = random.Random(5639)
        next_states = {}
        reward_map = {}

        for x in range(env.grid.width):
            for y in range(env.grid.height):
                cell = env.grid.get(x, y)
                if cell is None or cell.type != 'wall':
                    if 0 < x < env.grid.width and 0 < y < env.grid.height:
                        if cell is not None:
                            if cell.type == 'goal':
                                reward_map[(x,y)] = env.reward_spec['goal_reward']
                            elif cell.type == 'lava':
                                reward_map[(x,y)] = env.reward_spec['lava_reward']
                            elif cell.type == 'spiky_floor':
                                reward_map[(x,y)] = env.reward_spec['risky_tile_reward']
                        else:
                            reward_map[(x,y)] = 0 # TODO time_penalty here
                else:
                    reward_map[(x,y)] = 0
            
            # TODO add time_penalty on all states (also on starting state? --> mathematically correct?)

        self.env = env
        self.reward_map = reward_map
        self.next_states = next_states
    
    def transition_proba(
        self,
        state: tuple[int],
        action: int,
        next_state: tuple[int]
    ) -> float:
        assert 0 <= action < 4, "Illegal action"

        # return 0 from terminal states or walls
        current_state = self.env.grid.get(*state)
        if current_state is not None and \
            current_state.type in ['lava', 'goal', 'wall']:
            return 0 # TODO maybe return None

        x, y = state
        # index of ord_new_states is action that leads to that state
        ord_next_states = [(x-1,y), (x,y-1), (x+1,y), (x,y+1)]
        intended_state = ord_next_states[action]

        # return 0 if next_state can't be transitioned to
        if next_state not in ord_next_states:
            return 0

        # compute candidates for slipping (exclude cells on which agent cannot be)
        slip_candidates = 4
        for element in ord_next_states:
            next = self.env.grid.get(*element)
            if next is not None and not next.can_overlap():
                slip_candidates -= 1

        # if agent is surrounded by walls, avoid zero-division
        if slip_candidates == 0:
            return 0
        
        # slipping probability correction if agent slips in intended direction
        slip_factor = (slip_candidates - 1) / slip_candidates
        intended_proba = 1 - self.slip_proba * slip_factor

        if next_state == intended_state:
            return intended_proba
        else:
            return self.slip_proba/slip_candidates
    
    def make_random_policy(self):
        """Make random policy for instance's environment

        Returns:
            dict: The positional state-action mapping
        """        
        # init random state
        policy_dict = {}
        for x in range(self.env.grid.width):
            for y in range(self.env.grid.height):
                policy_dict[(x,y)] = self.rnd.randint(0, 3)
        return policy_dict
    
    def visualize_policy(self, policy_dict):
        grid = self.env.grid
        visual_policy = ""

        ansi_color = lambda code, text:  f"\33[{code}m{text}\33[0m"

        for i in range(grid.width):
            visual_policy += " " + str(i) + "  "
            if i == grid.width - 1:
                visual_policy += "\n"

        for y in range(grid.height):
            for x in range(grid.width):
                tile = grid.get(x, y)
                
                if tile is None:

                    dir_mapping = {0 : "<", 1 : "^", 2 : ">", 3 : "v"}
                    action = policy_dict[(x,y)]
                    dir_str = dir_mapping[action]

                    visual_policy += f"[{dir_str}] "
                elif tile.type == "wall":
                    w = ansi_color(36, "#")
                    visual_policy += f"[{w}] "
                elif tile.type == "lava":
                    l = ansi_color(41, "~")
                    visual_policy += f"[{l}] "
                elif tile.type == "goal":
                    g = ansi_color(42, "x")
                    visual_policy += f"[{g}] "
                
                if x == grid.width - 1: 
                    visual_policy += f" {y} \n"
                
        return visual_policy
    
    @classmethod
    def convert_to_policy(cls, sb3_model: str):
        """Convert a stable-baselines model to policy dict

        Args:
            sb3_model (str): path to model directory
        """
        policy_dict = {}

        model, env, rgb_on, render_size = model_env_from_path(sb3_model)
        _, second_env, _, _ = model_env_from_path(sb3_model)

        # no wrapping is needed, observations taken directly
        env.reset()
        grid = env.grid

        for y in range(grid.height):
            for x in range(grid.width):
                tile = grid.get(x, y)
                
                if tile is None or tile.type == 'spiky_floor':
                    env.unwrapped.agent_pos = (x, y)
                    if rgb_on:
                        obs = env.render(
                            mode="rgb_array",
                            highlight=False,
                            tile_size=render_size
                        )
                    else:
                        obs = env.tensor_obs()

                    action = int(model.predict(obs, deterministic=True)[0])
                    policy_dict[(x, y)] = action
                else: policy_dict[(x, y)] = 0 # action does not matter, just to fill a value
        return policy_dict, cls(second_env)

    def evaluate_policy(self, policy_dict, threshold=1e-10, beautify=1e-4):
        """Compute state-value function given a positional policy dictionary
        (mapping from grid-position to action).
        Algorithm: 'Iterative policy evalutation'

        Args:
            policy_dict (dict): The state-action mapping
            threshold (float): termination condition for value accuracy
            beautify (bool): Whether or not to set extremely small values to 0

        Returns:
            _type_: _description_
        """        
        values_old = {}
        values_new = {}
        grid = self.env.grid

        # initialize randomly with terminal values set to 0
        for x in range(grid.width):
            for y in range(grid.height):
                state = grid.get(x,y)
                if state is not None and \
                    state.type in ['goal', 'lava', 'wall']:
                    values_old[(x, y)] = 0
                else:
                    values_old[(x, y)] = self.rnd.random()

        diff = 0
        done = False
        while not done:
            diff = 0 # delta reset to 0 before looping over all states
            for key in policy_dict:
                x, y = key
                summands = []
                for next in [(x-1,y), (x,y-1), (x+1,y), (x,y+1)]:
                    if not (0 < next[0] < grid.width and 0 < next[1] < grid.height):
                        continue
                    r = self.reward_map[next]
                    v_new = values_old[next]
                    p = self.transition_proba(key, policy_dict[key], next)
                    summands.append(p * (r + self.gamma * v_new))
                values_new[key] = sum(summands)
                diff = max(diff, abs(values_new[key] - values_old[key]))
            done = True if diff < threshold else False
            values_old = values_new.copy()
        
        if beautify is not None:
            for key in values_new:
                if abs(values_new[key]) < beautify:
                    values_new[key] = 0 

        return values_new

    def compute_optimal_policy(self, current_policy=None, optimality_threshold=1e-6):
        """Compute an optimal policy based on iterative policy evaluation.
        An existing policy can be passed as a starting point.
        Algorithm: Policy Iteration

        Note: This algorithm is mainly inspired by Sutton & Barto's Book
        'Introduction to Reinforcement Learning'. The pseudocode can
        be found in Chapter '4.3 - Policy Iteration'. The pseudocode must
        be slightly adapted to avoid oscillation between equally good (optimal)
        policies as stated in exercise 4.4. Furthermore, this algorithm
        makes use of iterative policy evaluation, which is proven to converge to
        the correct value for a given policy. However, given that the actual
        algorithm cannot run indefinitely, one must stop at an estimation of
        the value. This fact implies that there are no guarantees that the 
        evalutation algorithm also returns the exact same value for two equally
        good (but different) policies, which would be the stopping rule for
        policy iteration. Therefore, I use an optimality_threshold, which stops
        policy iteration once two policies are 'similar enough' in their value
        function.
        """        
        def _argmax_action(from_state):
            """Return maximizing action given list of successor states
            and value function
            """
            grid = self.env.grid
            x, y = from_state
            for a in range(4):
                summands = []
                for next in [(x-1,y), (x,y-1), (x+1,y), (x,y+1)]:
                    if not (0 < next[0] < grid.width and 0 < next[1] < grid.height):
                        continue
                    r = self.reward_map[next]
                    v_next = value[next]
                    p = self.transition_proba(from_state, a, next)
                    summands.append(p * (r + self.gamma * v_next))
                if (sum(summands) - value[from_state]) > optimality_threshold:
                    return a
            return None
                

        # initialize random policy
        if current_policy is not None:
            policy = current_policy
        else:
            policy = self.make_random_policy()
        
        policy_stable = False
        while not policy_stable:
            policy_stable = True
            value = self.evaluate_policy(policy)
            for state in policy:
                maximizing_action = _argmax_action(state)
                if maximizing_action is not None:
                    policy[state] = maximizing_action
                    policy_stable = False
        return policy



# TESTING -----
# Method only used for testing the example in my thesis
def test_convergence_simple_env():
    lava_posis = []
    for x in range(1, 6):
        for y in range(1,6):
            if x == 3:
                continue
            lava_posis.append((x, y))

    kw = {
        "width" : 7,
        "height" : 7,
        "slip_proba" : 0.1,
        "agent_start_pos" : (3, 4),
        "goal_positions" : [(3, 2)],
        "lava_positions" : lava_posis
    }

    env: RiskyPathEnv = gym.make("MiniGrid-RiskyPath-v0", **kw)

    # add walls
    for y in range(1, 6):
        env.grid.set(2, y, Wall())
        env.grid.set(4, y, Wall())
    env.grid.set(3, 1, Wall())
    env.grid.set(3, 5, Wall())

    # add lava
    env.grid.set(2, 3, Lava())
    env.grid.set(4, 3, Lava())

    a = RiskyPathSolver(env)

    p = a.make_random_policy()
    p[(3,4)] = 1
    p[(3,3)] = 1
    print(a.visualize_policy(p))
    print(a.evaluate_policy(p)[(3,3)], a.evaluate_policy(p)[(3,4)])
    new_p = a.compute_optimal_policy()
    v_new = a.evaluate_policy(new_p)
    print(a.visualize_policy(new_p))
    print(v_new)
    # should be ~ 0.8969782085, 0.8880084

def test_slip_env_optimal_policy():
    a = RiskyPathSolver(gym.make("MiniGrid-RiskyPath-v0", slip_proba=0.1))

    p = a.make_random_policy()
    print("random policy:")
    print(a.visualize_policy(p))
    print("value of random policy:", a.evaluate_policy(p))
    print("optimal policy:")
    p = a.compute_optimal_policy()
    print(a.visualize_policy(p))
    print("value of optimal policy:", a.evaluate_policy(p))