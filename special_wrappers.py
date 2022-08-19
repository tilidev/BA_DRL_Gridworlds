from collections import defaultdict
import math
from typing import Optional
import gym
import numpy as np

from gym_minigrid.wrappers import TensorObsWrapper
# TODO Random Network Destillation IM wrapper(especially for rgb inputs)
# TODO Distributional Shift & Goal Misgeneralization Wrapper

class IntrinsicMotivationWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        total_steps: int,
        remove_extrinsic: bool = True,
        stop_after_n_steps: Optional[int] = 50_000,
        stop_after_progress: Optional[float] = None
    ):
        """A count-based intrinsic motivation rewards wrapper. Intrinsic rewards
        are computed based on state novelty by counting the occurences of
        recurrent observations. This wrapper should only be used on a wrapped
        version of 'RiskyPathEnv'

        Args:
            env: The wrapped environment (TensorObsWrapper)
            total_steps (int): _description_
            remove_extrinsic (bool): whether to return only intrinsic rewards
            stop_after_n_steps (Optional[int], optional): _description_
            stop_after_progress (Optional[float]): _description_
        """    
        # NOTE
        # CAREFUL, at the moment the environment must first be wrapped by
        # tensorwrapper and then by IM wrapper

        # check arguments
        absolute_stop = stop_after_n_steps is not None
        relative_stop = stop_after_progress is not None
        assert not relative_stop or (0 < stop_after_progress <= 1), \
            "'stop_after_progress' must be contained in open interval (0,1]"
        assert not (absolute_stop and relative_stop), \
            "Must choose between stopping after n steps or after"
        assert (absolute_stop or relative_stop), \
            "One of 'stop_after_n_steps' or 'stop_after_progress' must be set"
        assert isinstance(env, TensorObsWrapper), \
            "This wrapper should only be applied to a 'TensorObsWrapper'" \
            + "-wrapped 'RiskyPathEnv'-instance"

        super().__init__(env)

        if absolute_stop:
            self.final_step = stop_after_n_steps
        else:
            self.final_step = int(stop_after_progress * total_steps)

        self.learning_total_steps = total_steps
        self.remove_extrinsic = remove_extrinsic
        self.current_step = 0
        self.state_count_map = defaultdict(int)


    def step(self, action):
        """Acts like the normal step function but in an infinite horizon
        non-episodic setting. The environment does not return 'done==True'
        during pre-training. The agent will only transition to the starting
        position if the conditions in the original environment are met.
        This is to avoid leaking information of the reward model during 
        intrinsic motivation pre-training. (c.f. Burda et al., 2018)

        Args:
            action (int): The action taken in the environment

        Returns:
            observation, reward, done, info
        """
        # compute usual transition and increment current_step
        obs, extrinsic_reward, done, info = self.env.step(action)
        self.current_step += 1

        # return usual transition if wrapper should no longer be active
        if self.current_step >= self.final_step:
            if self.current_step == self.final_step:
                self._on_final_step_output()
            return obs, extrinsic_reward, done, info
        
        # compute empirical state-count based on the observed agent position
        current_state_n = self._map_pos_to_number(info["agent_pos"])
        self.state_count_map[current_state_n] += 1

        # compute empirical state-count reward bonus
        reward_bonus = 1 / math.sqrt(self.state_count_map[current_state_n])
        print(current_state_n)
        modified_reward = reward_bonus if self.remove_extrinsic \
            else extrinsic_reward + reward_bonus

        # return step() output
        if done and self.remove_extrinsic:
            # transition to starting state without ending episode
            # in infinite horizon setting
            transition_obs = self.env.reset()
            info_dict = {
                "agent_pos" : self.env.agent_pos,
                "previous_pos" : info["agent_pos"],
                "IM_wrapper_info" : "internal_reset"
            }
            return transition_obs, modified_reward, False, info_dict
        else:
            # return usual transition but with modified reward signal
            # if extrinsic rewards are also returned.
            # The episode ends in this case, as the intrinsic rewards are only
            # a bonus to the extrinsic rewards
            return obs, modified_reward, done, info


    def _map_pos_to_number(self, pos: tuple):
        """Hash a RiskyGridworld state into a number.
        This hash function is naive and expects stationary lava and goal
        tiles. The function only takes the agent's position as the hash
        inputs. If goal, lava or wall tiles change position, this function
        is no longer correctly applicable.

        Args:
            pos (tuple): the agent's position
        """
        agent_x, agent_y = pos
        hash_out = agent_y * self.env.grid.width + agent_x
        return hash_out


    def _on_final_step_output(self):
        """This method is called on the final pre_training step. Useful
        information is printed to stdout about the visited-states-distribution
        and other behavioral aspects.
        """
        grid_width = self.env.grid.width
        grid_height = self.env.grid.height
        out_text = ""
        print("\nState distribution during Pre-Training:")
        for y in range(grid_height):
            for x in range(grid_width):
                n = self.state_count_map[y * grid_width + x]
                n = str(round(n / self.current_step, 2))
                n += (4 - len(n)) * "0"
                if x == grid_width - 1:
                    out_text += f" [{n}]\n"
                else:
                    out_text += f" [{n}]"
        print(out_text + "\n")