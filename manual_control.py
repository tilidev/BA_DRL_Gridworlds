#!/usr/bin/env python3

# file taken from gym-minigrid @ Farama-Foundation and slightly modified

import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.envs.risky import ABSORBING_REWARD_GOAL, ABSORBING_REWARD_LAVA, ABSORBING_STATES, GOAL_REWARD, LAVA_REWARD, SPIKY_TILE_REWARD, STEP_PENALTY, RiskyPathEnv
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
from special_wrappers import IntrinsicMotivationWrapper, RandomizeGoalWrapper

is_RiskyPathEnv = False

def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

def step(action):
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        print('Resetting environment.')
        redraw(obs)
        time.sleep(0.2)
        reset()
    else:
        redraw(obs)

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        if is_RiskyPathEnv:
            action = env.new_actions.west
            step(action)
            return
        step(env.actions.left)
        return
    if event.key == 'right':
        if is_RiskyPathEnv:
            action = env.new_actions.east
            step(action)
            return
        step(env.actions.right)
        return
    if event.key == 'up':
        if is_RiskyPathEnv:
            action = env.new_actions.north
            step(action)
            return
        step(env.actions.forward)
        return
    if event.key == 'down':
        if is_RiskyPathEnv:
            step(env.new_actions.south)
            return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-RiskyPath-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw what the agent sees (partially observable view)",
    action='store_true'
)

# modifications by Tilio Schulze
parser.add_argument(
    "--spiky_active",
    default=False,
    help="if set, spiky tiles will be set",
    action='store_true'
)
parser.add_argument(
    "--wall_rebound",
    default=False,
    help="if set, the agent can rebound on walls",
    action='store_true'
)
parser.add_argument(
    "--slip_proba",
    default=0.,
    type=float,
    help="sets the agent's probability of slipping"
)
parser.add_argument(
    "--show_agent_dir",
    default=False,
    help="Whether or not the direction of the agent is to be shown",
    action="store_true"
)
parser.add_argument(
    "--wrap_IM",
    help="Will wrap environment with the 'IntrinsicMotivationWrapper'",
    action='store_true'
)
parser.add_argument(
    "--wrap_randomize",
    help="Will wrap environment with RandomizeGoalWrapper",
    action='store_true'
)
parser.add_argument(
    "--test_eval_randomizer",
    help="wrap env on the hard-coded goal randomization",
    action='store_true'
)

reward_model= {
    STEP_PENALTY : 0,
    GOAL_REWARD : 1,
    ABSORBING_STATES : False,
    ABSORBING_REWARD_GOAL : 0,
    ABSORBING_REWARD_LAVA : 0,
    SPIKY_TILE_REWARD : 0,
    LAVA_REWARD : -1
}

args = parser.parse_args()

env = gym.make(
    args.env,
    spiky_active=args.spiky_active,
    wall_rebound=args.wall_rebound,
    slip_proba=args.slip_proba,
    reward_spec=reward_model,
    show_agent_dir=args.show_agent_dir
)

is_RiskyPathEnv = True if "RiskyPath" in args.env else False

if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

if args.wrap_IM:
    env = TensorObsWrapper(env)
    env = IntrinsicMotivationWrapper(env, 100, stop_after_n_steps=30)

if args.wrap_randomize:
    env = RandomizeGoalWrapper(env, randomization=0.5)

if args.test_eval_randomizer:
    env = RandomizeGoalWrapper(env, eval_mode=True)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)