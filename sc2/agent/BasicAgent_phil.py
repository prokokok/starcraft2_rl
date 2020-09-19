# https://chatbotslife.com/building-a-basic-pysc2-agent-b109cde1477c

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random
import time


class ProtossBasicAgent(base_agent.BaseAgent):
    def __init__(self):
        super(ProtossBasicAgent, self).__init__()

        self.base_top_left = None
        self.pylon_built = False
        self.gateway_built = False
        self.gateway_rallied = False
        self.army_rallied = False

    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

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
        super(ProtossBasicAgent, self).step(obs)

        # time.sleep(0.5)

        if obs.first():
            self.base_top_left = None
            self.pylon_built = False
            self.gateway_built = False
            self.gateway_rallied = False
            self.army_rallied = False
            self.pylon_build = 0

            player_y, player_x = (
                    obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        # check_pylon = self.get_units_by_type(obs, units.Protoss.Pylon)
        # if check_pylon:
        #     self.pylon_build = check_pylon[0].build_progress
        pylons = self.get_units_by_type(obs, units.Protoss.Pylon)
        pylon_built = [True if pylon.build_progress == 100 else False for pylon in pylons]

        # self.army_rallied = self.gateway_built and self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id)

        if not self.pylon_built:
            if self.unit_type_is_selected(obs, units.Protoss.Probe):
                if self.can_do(obs, actions.FUNCTIONS.Build_Pylon_screen.id):

                    nexus = self.get_units_by_type(obs, units.Protoss.Nexus)
                    if len(nexus) > 0:
                        mean_x, mean_y = self.getMeanLocation(nexus)
                        target = self.transformLocation(int(mean_x), 0, int(mean_y), 30)
                        self.pylon_built = True

                        return actions.FUNCTIONS.Build_Pylon_screen("now", target)

            probes = self.get_units_by_type(obs, units.Protoss.Probe)
            if len(probes) > 0:
                probe = random.choice(probes)
                return actions.FUNCTIONS.select_point("select", (probe.x,
                                                                 probe.y))
        elif not self.gateway_built and any(pylon_built):
            if self.unit_type_is_selected(obs, units.Protoss.Probe):
                if self.can_do(obs, actions.FUNCTIONS.Build_Pylon_screen.id):
                    nexus = self.get_units_by_type(obs, units.Protoss.Nexus)
                    pylons = self.get_units_by_type(obs, units.Protoss.Pylons)

                    if len(nexus) > 0 and len(pylons) == 0:
                        mean_x, mean_y = self.getMeanLocation(nexus)
                        target = self.transformLocation(int(mean_x), 0, int(mean_y), 30)

                        return actions.FUNCTIONS.Build_Pylon_screen("now", target)
                    elif len(pylons) > 0:
                        pylon_coordinate = []
                        for pylon in pylons:
                            pylon_coordinate.append((pylon.x, pylon.y))
                            x_coordinate, y_coordinate = max(pylon_coordinate, key=lambda t: t[1])
                            target = self.transformLocation(int(x_coordinate), -10, int(y_coordinate), 0)
                        return actions.FUNCTIONS.Build_Pylon_screen("now", target)

                # if self.can_do(obs, actions.FUNCTIONS.Build_Gateway_screen.id):
                #     pylon = self.get_units_by_type(obs, units.Protoss.Pylon)
                #     if len(pylon) > 0:
                #         mean_x, mean_y = self.getMeanLocation(pylon)
                #
                #         if self.base_top_left:
                #             target = self.transformLocation(int(mean_x), -10, int(mean_y), 0)
                #         else:
                #             target = self.transformLocation(int(mean_x), -10, int(mean_y), 0)
                #
                #         self.gateway_built = True
                #         # return actions.FUNCTIONS.Build_Gateway_screen("now", target)
                #         return actions.FUNCTIONS.Build_Gateway_screen("now", target)

            probes = self.get_units_by_type(obs, units.Protoss.Probe)
            if len(probes) > 0:
                probe = random.choice(probes)
                return actions.FUNCTIONS.select_point("select", (probe.x,
                                                                 probe.y))

        elif not self.gateway_rallied:
            if self.unit_type_is_selected(obs, units.Protoss.Gateway):
                self.gateway_rallied = True

                if self.base_top_left:
                    return actions.FUNCTIONS.Rally_Units_minimap("now", [29, 21])
                else:
                    return actions.FUNCTIONS.Rally_Units_minimap("now", [29, 46])
            gateways = self.get_units_by_type(obs, units.Protoss.Gateway)
            if len(gateways) > 0:
                gateway = random.choice(gateways)
                return actions.FUNCTIONS.select_point("select", (gateway.x,
                                                                 gateway.y))
        elif obs.observation.player.food_cap - obs.observation.player.food_used - 1:
            if self.can_do(obs, actions.FUNCTIONS.Train_Zealot_quick.id):
                return actions.FUNCTIONS.Train_Zealot_quick("queued")

        # elif self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
        #
        #     if self.base_top_left:
        #         return actions.FUNCTIONS.Attack_minimap("now", [39, 45])
        #     else:
        #         return actions.FUNCTIONS.Attack_minimap("now", [21, 24])
        #
        #     if self.can_do(obs, actions.FUNCTIONS.select_army.id):
        #         return actions.FUNCTIONS.select_army("select")

        elif not self.army_rallied:
            if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                self.army_rallied = True

                if self.base_top_left:
                    return actions.FUNCTIONS.Attack_minimap("now", [39, 45])
                else:
                    return actions.FUNCTIONS.Attack_minimap("now", [21, 24])

            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")

        return actions.FUNCTIONS.no_op()

def main(unused_argv):
    #agent = ZergBasicAgent()
    # agent = TerranBasicAgent()
    agent = ProtossBasicAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                    #map_name="AbyssalReef",
                    map_name="Simple64",
                    #players=[sc2_env.Agent(sc2_env.Race.zerg),
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
