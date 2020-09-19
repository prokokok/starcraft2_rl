import random
import time
import math
import os.path

import numpy as np
import pandas as pd


from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app

DATA_FILE = 'rlagent_learning_data'

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_PROBE = 'selectprobe'
ACTION_BUILD_PYLON = 'buildpylon'
ACTION_BUILD_GATEWAY = 'buildgateways'
ACTION_SELECT_GATEWAY = 'selectgateways'
ACTION_BUILD_ZEALOT = 'buildzealot'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_PROBE,
    ACTION_BUILD_PYLON,
    ACTION_BUILD_GATEWAY,
    ACTION_SELECT_GATEWAY,
    ACTION_BUILD_ZEALOT,
    ACTION_SELECT_ARMY,
    ACTION_ATTACK,
]

#for mm_x in range(0, 64):
#    for mm_y in range(0, 64):
#        smart_actions.append(ACTION_ATTACK + '_' + str(mm_x) + '_' + str(mm_y))

for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 16 == 0 and (mm_y + 1) % 16 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))

KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5


# reference from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
# reference from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # choose best action
            # state_action = self.q_table.ix[observation, :]
            state_action = self.q_table.loc[observation, :]

            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        # q_predict = self.q_table.ix[s, a]
        q_predict = self.q_table.loc[s, a]
        # q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        q_target = r + self.gamma * self.q_table.loc[s_, :].max()

        # update
        # self.q_table.ix[s, a] += self.lr * (q_target - q_predict)
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class ProtossRLAgent(base_agent.BaseAgent):
    def __init__(self):
        super(ProtossRLAgent, self).__init__()

        self.base_top_left = None
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0

        self.previous_action = None
        self.previous_state = None

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]

        return [x, y]

    def getMeanLocation(self, unitList):
        sum_x = 0
        sum_y = 0
        for unit in unitList:
            sum_x += unit.x
            sum_y += unit.y
        mean_x = sum_x / len(unitList)
        mean_y = sum_y / len(unitList)

        return [mean_x, mean_y]

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
                obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and
                obs.observation.multi_select[0].unit_type == unit_type):
            return True

        return False

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def step(self, obs):
        super(ProtossRLAgent, self).step(obs)

        # time.sleep(0.5)

        if obs.last():
            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

        if obs.first():
            player_y, player_x = (
                        obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        pylons = self.get_units_by_type(obs, units.Protoss.Pylon)
        pylon_built = [True if pylon.build_progress == 100 else False for pylon in pylons]

        pylon_count = len(self.get_units_by_type(obs, units.Protoss.Pylon))
        gateway_count = len(self.get_units_by_type(obs, units.Protoss.Gateway))

        supply_limit = obs.observation.player.food_cap
        army_supply = obs.observation.player.food_used

        killed_unit_score = obs.observation.score_cumulative.killed_value_units
        killed_building_score = obs.observation.score_cumulative.killed_value_structures

        # current_state = np.zeros(5000)
        # current_state[0] = pylon_count
        # current_state[1] = gateway_count
        # current_state[2] = supply_limit
        # current_state[3] = army_supply

        # hot_squares = np.zeros(4096)
        # enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.ENEMY).nonzero()
        # for i in range(0, len(enemy_y)):
        #     y = int(enemy_y[i])
        #     x = int(enemy_x[i])
        #
        #     hot_squares[((y - 1) * 64) + (x - 1)] = 1
        #
        # if not self.base_top_left:
        #     hot_squares = hot_squares[::-1]
        #
        # for i in range(0, 4096):
        #     current_state[i + 4] = hot_squares[i]

        current_state = np.zeros(20)
        current_state[0] = pylon_count
        current_state[1] = gateway_count
        current_state[2] = supply_limit
        current_state[3] = army_supply

        hot_squares = np.zeros(16)
        enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.ENEMY).nonzero()
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 16))
            x = int(math.ceil((enemy_x[i] + 1) / 16))

            hot_squares[((y - 1) * 4) + (x - 1)] = 1

        if not self.base_top_left:
            hot_squares = hot_squares[::-1]

        for i in range(0, 16):
            current_state[i + 4] = hot_squares[i]

        if self.previous_action is not None:
            reward = 0

            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD

            if killed_building_score > self.previous_killed_building_score:
                reward += KILL_BUILDING_REWARD

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]

        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_state = current_state
        self.previous_action = rl_action

        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        if smart_action == ACTION_DO_NOTHING:
            return actions.FUNCTIONS.no_op()

        elif smart_action == ACTION_SELECT_PROBE:
            if self.can_do(obs, actions.FUNCTIONS.select_point.id):
                probes = self.get_units_by_type(obs, units.Protoss.Probe)
                if len(probes) > 0:
                    probe = random.choice(probes)
                    if probe.x >= 0 and probe.y >= 0:
                        return actions.FUNCTIONS.select_point("select", (probe.x,
                                                                         probe.y))
        elif smart_action == ACTION_BUILD_PYLON:
            if self.can_do(obs, actions.FUNCTIONS.Build_Pylon_screen.id):
                nexus = self.get_units_by_type(obs, units.Protoss.Nexus)
                pylons = self.get_units_by_type(obs, units.Protoss.Pylon)

                if len(nexus) > 0 and len(pylons) == 0:
                    mean_x, mean_y = self.getMeanLocation(nexus)
                    target = self.transformDistance(int(mean_x), -20, int(mean_y), 30)

                    return actions.FUNCTIONS.Build_Pylon_screen("now", target)
                elif len(pylons) > 0:
                    pylon_coordinate = []
                    for pylon in pylons:
                        pylon_coordinate.append((pylon.x, pylon.y))
                    x_coordinate, y_coordinate = max(pylon_coordinate, key=lambda t: t[1])
                    target = self.transformDistance(int(x_coordinate), 10, int(y_coordinate), 0)

                    return actions.FUNCTIONS.Build_Pylon_screen("now", target)

        elif smart_action == ACTION_BUILD_GATEWAY:
            if self.can_do(obs, actions.FUNCTIONS.Build_Gateway_screen.id):

                pylons = self.get_units_by_type(obs, units.Protoss.Pylon)
                pylon_built = [True if pylon.build_progress == 100 else False for pylon in pylons]

                avail_pylons = []
                for pylon_index in range(0,len(pylons)):
                    # print(pylon_index)
                    if pylon_built[pylon_index]:
                        avail_pylons.append(pylons[pylon_index])

                # target_pylon = len(avail_pylons)
                # print(target_pylon)

                target_pylon = random.choice(avail_pylons)
                target = self.transformDistance(int(target_pylon.x), 0, int(target_pylon.y), -10)

                return actions.FUNCTIONS.Build_Gateway_screen("now", target)

        elif smart_action == ACTION_SELECT_GATEWAY:
            if self.can_do(obs, actions.FUNCTIONS.select_point.id):
                gateways = self.get_units_by_type(obs, units.Protoss.Gateway)
                if len(gateways) > 0:
                    gateway = random.choice(gateways)
                    if gateway.x >= 0 and gateway.y >= 0:
                        return actions.FUNCTIONS.select_point("select", (gateway.x,
                                                                         gateway.y))

        elif smart_action == ACTION_BUILD_ZEALOT:
            if self.can_do(obs, actions.FUNCTIONS.Train_Zealot_quick.id):
                return actions.FUNCTIONS.Train_Zealot_quick("queued")

        elif smart_action == ACTION_SELECT_ARMY:
            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")

        elif smart_action == ACTION_ATTACK:
            # if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
            if not self.unit_type_is_selected(obs, units.Protoss.Probe) and self.can_do(obs,
                                                                                        actions.FUNCTIONS.Attack_minimap.id):
                return actions.FUNCTIONS.Attack_minimap("now", self.transformLocation(int(x), int(y)))

        return actions.FUNCTIONS.no_op()

def main(unused_argv):
    agent = ProtossRLAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                    # map_name="AbyssalReef",
                    map_name="Simple64",
                    # players=[sc2_env.Agent(sc2_env.Race.zerg),
                    players=[sc2_env.Agent(sc2_env.Race.protoss),
                             sc2_env.Bot(sc2_env.Race.random,
                                         sc2_env.Difficulty.very_easy)],
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(screen=84, minimap=64),
                        use_feature_units=True),
                    step_mul=1,
                    game_steps_per_episode=0,
                    visualize=True) as env:

                agent.setup(env.observation_spec(), env.action_spec())

                timesteps = env.reset()
                agent.reset()

                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)