from collections import defaultdict
import math
from typing import Optional, Union
import gym
from gym.utils import seeding
from gym_minigrid.envs.risky import RiskyPathEnv

from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from gym_minigrid.wrappers import TensorObsWrapper
# NOTE implement "Random Network Destillation" IM wrapper (for rgb inputs)

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
            total_steps (int): the amount of steps used for learning
            remove_extrinsic (bool): whether to return only intrinsic rewards
            stop_after_n_steps (Optional[int]): number of pretraining steps
            stop_after_progress (Optional[float]): percentage of pretraining
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
        self.state_count_map = defaultdict(int) # do not rename


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


    def _map_pos_to_number(self, pos: tuple) -> int:
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


class RandomizeGoalWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        randomization: float = 0.02,
        eval_mode: bool = False
    ):
        """Creates a self-shifting environment for training agent robustness
        and avoid goal misgeneralization. Will randomize placement of
        reward-inducing tiles. This wrapper should not be applied with the
        intrinsic reward wrapper, as it assumes a stationary state distribution
        being dependent only on the agent's position.

        Args:
            env: the environment to be randomized
            randomization: The ratio of episodes with randomization
            eval_mode: evaluate agent on hand-picked goal locations
        """
        if isinstance(env, VecTransposeImage):
            # access the VecEnv's underlying true environment
            env = env.envs[0]

        assert isinstance(env.unwrapped, RiskyPathEnv), \
            "Must be RiskyPathEnv"
        assert 0 < randomization <= 1, \
            "Randomization factor must be in interval (0,1]"
        assert not hasattr(env, "state_count_map"), \
            "Should not apply this wrapper with Intrinsic Motivation wrapper"
        
        self.is_eval = eval_mode
        if self.is_eval:
            self.eval_idx = 0
            self.eval_goals = [(1,3), (3,5), (7,5), (9,9), (9,1), (5,8)]
        self.randomization = randomization

        # get originial goal positions (shallow copy)
        self._original_goal_positions = env.goal_positions.copy()

        # set available tiles for random goal placement
        self._reserved_positions = []
        self._reserved_positions.extend(env.lava_positions)
        self._reserved_positions.append(tuple(env.agent_start_pos))

        self.is_modified = False
        super().__init__(env)

    def reset(self):
        if self.is_eval:
            setattr(
                self.env.unwrapped,
                'goal_positions',
                [self.eval_goals[self.eval_idx]]
            )
            l = len(self.eval_goals) - 1
            self.eval_idx = self.eval_idx + 1 if self.eval_idx < l else 0
        elif self.env.np_random.random() < self.randomization:
            # set new environment goal tiles
            # Access must be to unwrapped env due to the way gym.Wrapper works
            setattr(
                self.env.unwrapped,
                'goal_positions',
                self._generate_new_goals()
            )
            self.is_modified = True
            return self.env.reset()
        elif self.is_modified:
            self.env.unwrapped.goal_positions = self._original_goal_positions
            self.is_modified = False

        return super().reset()
    
    def step(self, action):
        return super().step(action)
    
    def _generate_new_goals(self):
        width, height = self.env.width, self.env.height
        all_positions = [] # fill with positions inside walls
        for i in range(1, width - 1):
            all_positions.extend([(i, j) for j in range(1, height - 1)])
        reserved_positions = self._reserved_positions.copy()
        for p in reserved_positions:
            all_positions.remove(p)

        goal_count = len(self._original_goal_positions)
        new_positions = []
        for _ in range(goal_count):
            new_goal_idx = self.env.np_random.choice(len(all_positions))
            new_goal = all_positions[new_goal_idx]
            assert new_goal not in reserved_positions
            all_positions.pop(new_goal_idx)
            new_positions.append(new_goal)

        return new_positions