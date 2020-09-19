import random
import time
import math
import os.path

import numpy as np
import pandas as pd


from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features, units
from absl import app

import torch
from torch.utils.tensorboard import SummaryWriter

from skdrl.pytorch.model.mlp import NaiveMultiLayerPerceptron
from skdrl.pytorch.model.naivedqn import NaiveDQN
from skdrl.pytorch.util.train_util import EMAMeter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

class ProtossAgentWithRawActsAndRawObs(base_agent.BaseAgent):
    actions = ("do_nothing",
               "harvest_minerals",
               "build_pylons",
               "build_gateways",
               "train_zealot",
               "attack")

    def get_my_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.ENEMY]

    def get_my_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.ENEMY]

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def getMeanLocation(self, unitList):
        sum_x = 0
        sum_y = 0
        for unit in unitList:
            sum_x += unit.x
            sum_y += unit.y
        mean_x = sum_x / len(unitList)
        mean_y = sum_y / len(unitList)

        return [mean_x, mean_y]

    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def step(self, obs):
        super(ProtossAgentWithRawActsAndRawObs, self).step(obs)
        if obs.first():
            nexus = self.get_my_units_by_type(
                obs, units.Protoss.Nexus)[0]
            self.base_top_left = (nexus.x < 32)

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def harvest_minerals(self, obs):
        probes = self.get_my_units_by_type(obs, units.Protoss.Probe)
        idle_probes = [probe for probe in probes if probe.order_length == 0]
        if len(idle_probes) > 0:
            mineral_patches = [unit for unit in obs.observation.raw_units
                               if unit.unit_type in [
                                   units.Neutral.BattleStationMineralField,
                                   units.Neutral.BattleStationMineralField750,
                                   units.Neutral.LabMineralField,
                                   units.Neutral.LabMineralField750,
                                   units.Neutral.MineralField,
                                   units.Neutral.MineralField750,
                                   units.Neutral.PurifierMineralField,
                                   units.Neutral.PurifierMineralField750,
                                   units.Neutral.PurifierRichMineralField,
                                   units.Neutral.PurifierRichMineralField750,
                                   units.Neutral.RichMineralField,
                                   units.Neutral.RichMineralField750
                               ]]
            probe = random.choice(idle_probes)
            distances = self.get_distances(obs, mineral_patches, (probe.x, probe.y))
            mineral_patch = mineral_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                "now", probe.tag, mineral_patch.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_pylons(self, obs):
        nexus = self.get_my_units_by_type(obs, units.Protoss.Nexus)
        pylons = self.get_my_units_by_type(obs, units.Protoss.Pylon)
        probes = self.get_my_units_by_type(obs, units.Protoss.Probe)

        if len(nexus) == 1 and len(pylons) == 0 and obs.observation.player.minerals >= 100:
            mean_x, mean_y = self.getMeanLocation(nexus)
            pylon_xy = self.transformDistance(int(mean_x), -20, int(mean_y), 30)
            distances = self.get_distances(obs, probes, pylon_xy)
            probe = probes[np.argmin(distances)]

            return actions.RAW_FUNCTIONS.Build_Pylon_pt(
                "now", probe.tag, pylon_xy)
        elif len(pylons) > 0:
            pylon_coordinate = []
            for pylon in pylons:
                pylon_coordinate.append((pylon.x, pylon.y))
            x_coordinate, y_coordinate = max(pylon_coordinate, key=lambda t: t[1])
            pylon_xy = self.transformDistance(int(x_coordinate), 10, int(y_coordinate), 0)
            distances = self.get_distances(obs, probes, pylon_xy)
            probe = probes[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.FUNCTIONS.Build_Pylon_pt("now", probe.tag, pylon_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_gateways(self, obs):
        completed_pylons = self.get_my_completed_units_by_type(obs, units.Protoss.Pylon)
        gateways = self.get_my_units_by_type(obs, units.Protoss.Gateway)
        probes = self.get_my_units_by_type(obs, units.Protoss.Probe)

        if (len(completed_pylons) > 0 and len(gateways) == 0 and
                obs.observation.player.minerals >= 150 and len(probes) > 0):
            target_pylon = random.choice(completed_pylons)
            gateway_xy = self.transformDistance(int(target_pylon.x), 0, int(target_pylon.y), -10)
            distances = self.get_distances(obs, probes, gateway_xy)
            probe = probes[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Gateway_pt("now", probe.tag, gateway_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def train_marine(self, obs):
        completed_gateways = self.get_my_completed_units_by_type(
            obs, units.Protoss.Gateway)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_gateways) > 0 and obs.observation.player.minerals >= 100
                and free_supply > 1):
            gateways = self.get_my_units_by_type(obs, units.Protoss.Gateway)
            gateway = random.choice(gateways)
            if gateway.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Zealot_quick("now", gateway.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def attack(self, obs):
        zealots = self.get_my_units_by_type(obs, units.Protoss.Zealot)
        if len(zealots) > 0:
            attack_xy = (38, 44) if self.base_top_left else (19, 23)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            zealots_tags = [zealot.tag for zealot in zealots]

            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", zealots_tags, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

class ProtossRandomAgent(ProtossAgentWithRawActsAndRawObs):
    def step(self, obs):
        super(ProtossRandomAgent, self).step(obs)
        action = random.choice(self.actions)
        return getattr(self, action)(obs)

class ProtossRLAgentWithRawActsAndRawObs(ProtossAgentWithRawActsAndRawObs):
    def __init__(self):
        super(ProtossRLAgentWithRawActsAndRawObs, self).__init__()

        self.s_dim = 21
        self.a_dim = 6
        self.lr = 1e-4
        self.gamma = 1.0
        self.epsilon = 1.0

        self.qnetwork = NaiveMultiLayerPerceptron(input_dim=self.s_dim,
                           output_dim=self.a_dim,
                           num_neurons=[128],
                           hidden_act_func='ReLU',
                           out_act_func='Identity').to(device)

        self.data_file = 'rlagent_with_naive_dqn'
        if os.path.isfile(self.data_file + '.pt'):
            self.qnetwork.load_state_dict(torch.load(self.data_file + '.pt'))

        self.dqn = NaiveDQN(state_dim=self.s_dim,
                             action_dim=self.a_dim,
                             qnet=self.qnetwork,
                             lr=self.lr,
                             gamma=self.gamma,
                             epsilon=self.epsilon).to(device)

        self.print_every = 50
        self.ema_factor = 0.5
        self.ema = EMAMeter(self.ema_factor)
        self.cum_reward = 0
        self.cum_loss = 0
        self.episode_count = 0

        self.new_game()

    def reset(self):
        super(TerranRLAgentWithRawActsAndRawObs, self).reset()
        self.new_game()

    def new_game(self):
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None
        self.cum_reward = 0
        self.cum_loss = 0

    def get_state(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)

        queued_marines = (completed_barrackses[0].order_length
        if len(completed_barrackses) > 0 else 0)

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        can_afford_supply_depot = obs.observation.player.minerals >= 100
        can_afford_barracks = obs.observation.player.minerals >= 150
        can_afford_marine = obs.observation.player.minerals >= 100

        enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
        enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]
        enemy_command_centers = self.get_enemy_units_by_type(
            obs, units.Terran.CommandCenter)
        enemy_supply_depots = self.get_enemy_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_completed_supply_depots = self.get_enemy_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
        enemy_completed_barrackses = self.get_enemy_completed_units_by_type(
            obs, units.Terran.Barracks)
        enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)

        return (len(command_centers),
                len(scvs),
                len(idle_scvs),
                len(supply_depots),
                len(completed_supply_depots),
                len(barrackses),
                len(completed_barrackses),
                len(marines),
                queued_marines,
                free_supply,
                can_afford_supply_depot,
                can_afford_barracks,
                can_afford_marine,
                len(enemy_command_centers),
                len(enemy_scvs),
                len(enemy_idle_scvs),
                len(enemy_supply_depots),
                len(enemy_completed_supply_depots),
                len(enemy_barrackses),
                len(enemy_completed_barrackses),
                len(enemy_marines))

    def step(self, obs):
        super(TerranRLAgentWithRawActsAndRawObs, self).step(obs)

        #time.sleep(0.5)

        state = self.get_state(obs)
        state = torch.tensor(state).float().view(1, self.s_dim).to(device)
        action_idx = self.dqn.choose_action(state)
        action = self.actions[action_idx]
        done = True if obs.last() else False
        if self.previous_action is not None:
            loss = self.dqn.learn(self.previous_state.to(device),
                              torch.tensor(self.previous_action).view(1, 1).to(device),
                              torch.tensor(obs.reward).view(1, 1).to(device),
                              state.to(device),
                              torch.tensor(done).float().view(1, 1).to(device)
                              )
            self.cum_loss += loss.detach().numpy()
        self.cum_reward += obs.reward
        self.previous_state = state
        self.previous_action = action_idx

        if obs.last():
            self.episode_count = self.episode_count + 1
            torch.save(self.dqn.qnet.state_dict(), self.data_file + '.pt')

            self.ema.update(self.cum_reward)
            writer.add_scalar("Loss/online", self.cum_loss/obs.observation.game_loop, self.episode_count)
            writer.add_scalar("Score", self.ema.s, self.episode_count)

            if self.episode_count % self.print_every == 0:
                print("Episode {} || EMA: {} || EPS : {}".format(self.episode_count, self.ema.s, self.dqn.epsilon))

            if self.episode_count >= 150:
                self.dqn.epsilon *= 0.999

        return getattr(self, action)(obs)

# def main(unused_argv):
#    agent1 = TerranRLAgentWithRawActsAndRawObs()
#    agent2 = TerranRandomAgent()
#    try:
#        with sc2_env.SC2Env(
#                map_name="Simple64",
#                players=[sc2_env.Agent(sc2_env.Race.terran),
#                         sc2_env.Agent(sc2_env.Race.terran)],
#                agent_interface_format=features.AgentInterfaceFormat(
#                    action_space=actions.ActionSpace.RAW,
#                    use_raw_units=True,
#                    raw_resolution=64,
#                ),
#                step_mul=8,
#                disable_fog=True,
#        ) as env:
#            run_loop.run_loop([agent1, agent2], env, max_episodes=1000)
#    except KeyboardInterrupt:
#        pass


# def main(unused_argv):
#     agent = TerranRLAgentWithRawActsAndRawObs()
#     try:
#         with sc2_env.SC2Env(
#                 map_name="Simple64",
#                 players=[sc2_env.Agent(sc2_env.Race.terran),
#                          sc2_env.Bot(sc2_env.Race.terran,
#                                      sc2_env.Difficulty.very_easy)],
#                 agent_interface_format=features.AgentInterfaceFormat(
#                     action_space=actions.ActionSpace.RAW,
#                     use_raw_units=True,
#                     raw_resolution=64,
#                 ),
#                 step_mul=8,
#                 disable_fog=True,
#         ) as env:
#             agent.setup(env.observation_spec(), env.action_spec())
#
#             timesteps = env.reset()
#             agent.reset()
#
#             while True:
#                 step_actions = [agent.step(timesteps[0])]
#                 if timesteps[0].last():
#                     break
#                 timesteps = env.step(step_actions)
#     except KeyboardInterrupt:
#         pass

# def main(unused_argv):
#     agent = TerranRLAgentWithRawActsAndRawObs()
#     try:
#         while True:
#             with sc2_env.SC2Env(
#                     map_name="Simple64",
#                     players=[sc2_env.Agent(sc2_env.Race.terran),
#                              sc2_env.Bot(sc2_env.Race.terran,
#                                          sc2_env.Difficulty.very_easy)],
#                     agent_interface_format=features.AgentInterfaceFormat(
#                         action_space=actions.ActionSpace.RAW,
#                         use_raw_units=True,
#                         raw_resolution=64,
#                     ),
#                     step_mul=8,
#                     disable_fog=True,
#                     game_steps_per_episode=0,
#                     visualize=False) as env:
#
#               agent.setup(env.observation_spec(), env.action_spec())
#
#               timesteps = env.reset()
#               agent.reset()
#
#               while True:
#                   step_actions = [agent.step(timesteps[0])]
#                   if timesteps[0].last():
#                       break
#                   timesteps = env.step(step_actions)
#
#     except KeyboardInterrupt:
#         pass


def main(unused_argv):
   agent1 = TerranRLAgentWithRawActsAndRawObs()
   try:
       with sc2_env.SC2Env(
               map_name="Simple64",
               players=[sc2_env.Agent(sc2_env.Race.terran),
                        sc2_env.Bot(sc2_env.Race.terran,
                                    sc2_env.Difficulty.very_easy)],
               agent_interface_format=features.AgentInterfaceFormat(
                   action_space=actions.ActionSpace.RAW,
                   use_raw_units=True,
                   raw_resolution=64,
               ),
               step_mul=8,
               disable_fog=True,
               visualize=False
       ) as env:
           run_loop.run_loop([agent1], env, max_episodes=1000)
   except KeyboardInterrupt:
       pass

if __name__ == "__main__":
    app.run(main)
