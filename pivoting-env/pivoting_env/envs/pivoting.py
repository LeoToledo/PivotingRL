import numpy as np
from gym import utils
from pivoting_env.envs import mujoco_env
from pivoting_env.envs.controllers_utils import CtrlUtils
import os
import yaml

# Read YAML file
with open(f'{os.getcwd()}/parameters.yaml', 'r') as file_descriptor:
    parameters = yaml.load(file_descriptor)

MAX_EP_LEN = parameters['model']['max_ep_len']
N_JOINTS = 7


class PivotingEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):

        self.model_name = parameters['train']['model_name']

        # Setting initial parameters
        self.tool2gripper_angle = None
        self.tool2gripper2desired_angle = None
        self.desired_angle = 0
        self.grippers_angle = 0

        self.current_ep = 0
        self.counter = 0

        self.acceptable_error = 0
        self.current_step = 0
        self.drop = False

        self.ep_ret = 0
        self.ep_len = 0
        self.ep_ret_list = []

        self.ctrl = None

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'pivoting_kuka.xml', 8)

    def ctrl_action_torque(self, a, qposd_robot, method):
        '''Métodos de torque

        Args:
            method (int): Chooses among the 3 possible control methods
            a ():
            qposd_robot ():
        '''

        # método 1: nessa config o PPO vai atuar JUNTO com meu controlador em torque EM TODAS AS JUNTAS
        if method == 1:
            u = self.ctrl.ctrl_action(self.sim) + a[:N_JOINTS]

        # método 2: nessa config o PPO vai controlar tudo em torque sem ajuda do meu controlador
        elif method == 2:
            u = a[:N_JOINTS]

        # método 3: aqui o PPO vai atuar só nas juntas "planares" em torque
        elif (method == 3):
            u = np.zeros(8)
            u[0:7] = self.ctrl.ctrl_action(self.sim)
            u[parameters['model']['ppo_acting_joints']] += a[parameters['model']['ppo_acting_joints']]

        # método 4: aqui o PPO vai atuar integralmente nas juntas escolhidas #3, 5, 6
        elif (method == 4):
            u = np.zeros(8)
            u[0:7] = self.ctrl.ctrl_action(self.sim)
            u[parameters['model']['ppo_acting_joints']] = a[parameters['model']['ppo_acting_joints']]
        return u

    # def ctrl_action_position(self, a, method):
    #     '''Métodos de posição
    #
    #     Args:
    #         method (int): Chooses among the 3 possible control methods
    #         a ():
    #     '''
    #     #método 1: nessa config o PPO vai atuar JUNTO com meu controlador em posição em TODAS AS JUNTAS
    #     if method == 1:
    #         qposd_ppo = np.array([0, 0, 0, -np.pi / 2, -np.pi / 2, 0, 0]) + a[:N_JOINTS]
    #
    #     #método 2: nessa config o PPO vai controlar tudo em posição. Aprendizado mais lento!
    #     elif method == 2:
    #         qposd_ppo = a[:N_JOINTS]
    #
    #     #método 3: aqui o PPO vai atuar só nas juntas "planares" em posição
    #     elif method == 3:
    #         qposd_ppo = np.array([0, 0, 0, -np.pi / 2, -np.pi / 2, 0, 0])
    #         qposd_ppo[JOINT_5] += a[JOINT_5]
    #         qposd_ppo[JOINT_6] += a[JOINT_6]
    #
    #     # método 4: aqui o PPO vai atuar integralmente nas juntas escolhidas
    #     elif method == 4:
    #         qposd_ppo = np.array([0, 0, 0, -np.pi / 2, -np.pi / 2, 0, 0])
    #         qposd_ppo[JOINT_5] = a[JOINT_5]
    #         qposd_ppo[JOINT_6] = a[JOINT_6]

    # return qposd_ppo

    def calculate_reward(self, ob):
        """
        Calculates the immediate reward given the current observation data
        Args:
            ob (numpy.ndarray): Array with all observation space parameters

        Returns:
            ob (numpy.ndarray): Array with all observation space parameters
            reward (float): Immediate reward value
            done (bool): Indicates if the episode finished
        """
        self.current_step += 1

        # Checa se caiu no chão
        # if ob[2] != 0:
        #     reward = -10000
        #     done = 1
        #     self.drop = True
        #     return ob, reward, done, {}

        # print(f'Desired: {np.round(self.desired_angle, 1)} \t '
        #       f'Current: {np.round(self.tool2gripper_angle, 1)} \t'
        #       f'Related: {np.round(np.abs(self.tool2gripper2desired_angle), 1)} \t'
        #       f'Acceptable: {np.round(self.acceptable_error, 1)}')
        #
        # Checa se está na região de sucesso
        if np.abs(self.tool2gripper2desired_angle) < self.acceptable_error:

            reward = ((-1) * np.abs(self.tool2gripper2desired_angle) / np.abs(
                2 * 30)) - ob[2] / 2
            self.counter = self.counter + 1
            done = 0

            # Caso fique uma quantidade de tempo na região de sucesso
            if self.counter > parameters['model']['reward']['steps_to_converge']:
                reward = parameters['model']['reward']['of_sucess']
                # Zerando o contador de tempo na zona de sucesso
                self.counter = 0
                # Zerando o tempo do episódio
                self.ep_len = 0
                # Zerando a recompensa acumulada
                self.ep_ret = 0

                done = 1
                self.drop = False
                return ob, reward, done, {}

        # Se não completar
        else:
            self.counter = 0
            reward = ((-1) * np.abs(self.tool2gripper2desired_angle) / np.abs(
                2 * 30)) - ob[2] / 2
            done = 0

        # Reward total e duração do episodio
        self.ep_ret += reward
        self.ep_len += 1

        if (self.current_step == MAX_EP_LEN):
            # Zerando a duração do episodio
            self.ep_len = 0
            # Atualizando o episodio atual
            self.current_ep += 1
            # Zerando a recompensa acumulada
            self.ep_ret = 0

        self.drop = False
        return ob, reward, done, {}

    def step(self, a):
        """
        Args:
            a (numpy.ndarray): Array with all action space elements
        Returns:
            ob (numpy.ndarray): Array with all observation space parameters
            reward (float): Immediate reward value
            done (bool): Indicates if the episode finished
        """

        if self.ctrl is None:
            self.ctrl = CtrlUtils(self.sim)

        qposd_robot = np.array([0, 0, 0, -np.pi / 2, -np.pi / 2, 0, 0])

        # TODO: NAO USAR ESSES AQUI POR ENQUANTO, TEMOS QUE ALTERAR O XML
        # PPO ATUANDO NA POSICAO (comentar aqui se for usar torque)
        # qpos_ppo = self.ctrl_action_position(a, method=3)
        # self.ctrl.calculate_errors(self.sim, qpos_ref=qpos_ppo)
        # u = self.ctrl.ctrl_action(self.sim)

        # PPO ATUANDO NO TORQUE
        self.ctrl.calculate_errors(self.sim, qpos_ref=qposd_robot)
        u = self.ctrl_action_torque(a, qposd_robot=qposd_robot, method=parameters['model']['control_method'])

        # NAO EDITAR DAQUI PRA BAIXO
        self.sim.data.ctrl[0:self.ctrl.nv + 1] = u  # jogando ação de controle nas juntas

        try:
            self.sim.step()
        except:
            self.sim.data.ctrl[0:self.ctrl.nv + 1] = u / 5  # jogando ação de controle nas juntas
            self.sim.step()

        # Get observation
        ob = self._get_obs()

        # Calculates reward
        ob, reward, done, _ = self.calculate_reward(ob)

        return ob, reward, done, _

    def get_desired_angle(self):
        return self.desired_angle

    def get_current_angle(self):
        return self.tool2gripper_angle

    def get_drop_bool(self):
        if self.drop:
            return 1
        return 0

    @property
    def get_fail_obs(self):

        # Tool's angle
        obs = self.sim.data.get_joint_qpos("tool")
        tools_angle = np.arctan2(2 * (obs[3] * obs[6] + obs[4] * obs[5]), (1 - 2 * (obs[5] ** 2 + obs[6] ** 2)))
        tools_angle = 180 * tools_angle / np.pi

        # Forces over the part
        xmat = self.sim.data.get_body_xmat('tool_1')
        xmat = np.reshape(xmat, (9, 1)).ravel()

        # Gripper's angle
        grippers_angle = (-1) * (self.sim.data.get_joint_qpos("kuka_joint_6"))
        grippers_angle = 180 * grippers_angle / np.pi

        # Gripper's velocity
        grippers_vel = self.sim.data.get_joint_qvel("kuka_joint_6")

        # Gripper's position
        grippers_pos = self.sim.data.get_joint_qpos("kuka_joint_6")

        fail_obs = {'desired_angle': self.desired_angle, 'tools_angle': tools_angle, 'grippers_angle': grippers_angle,
                    'grippers_vel': grippers_vel, 'grippers_pos': grippers_pos}
        for i, value in enumerate(xmat):
            fail_obs[f'force_{i}'] = value

        return fail_obs

    def reset_model(self):
        # Recebe qpos e qvel do modelo do mujoco, com vários dados que não serão utilizados
        qpos_init_robot = [0, 0, 0, -np.pi / 2, -np.pi / 2, 0, 0]
        qpos_init_gripper = [0]
        qpos_init_tool = [0.04, 0, 1.785, 1, 0, 0, 0]
        qpos = np.concatenate((qpos_init_robot, qpos_init_gripper, qpos_init_tool))
        qvel = self.init_qvel

        # Atualizando o numero do episódio e steps atuais. Está com gambiarra
        self.counter = 0
        self.current_step = 0

        self.desired_angle = self._set_degree_range()

        # Definindo o acceptable_error aceitável para a conclusão do objetivo
        acceptable_error_percentage = parameters['model']['acceptable_error_percentage']
        max_acceptable_error = parameters['model']['max_acceptable_error']
        self.acceptable_error = min(np.abs(self.desired_angle * acceptable_error_percentage), max_acceptable_error)

        if self.desired_angle <= 6 or self.desired_angle >= -6:
            self.acceptable_error = self.acceptable_error + 0.3

        self.set_state(qpos, qvel)
        return self._get_obs()

    # Sets the range of desired angles according to the current model
    def _set_degree_range(self):

        model_name = self.model_name

        if model_name == 'pivoting_25_30':
            return np.random.choice([-30, -29, -28, -27, -26, 26, 27, 28, 29, 30])
        elif model_name == 'pivoting_20_25':
            return np.random.choice([-25, -24, -23, -22, -21, 21, 22, 23, 24, 25])
        elif model_name == 'pivoting_15_20':
            return np.random.choice([-20, -19, -18, -17, -16, 16, 17, 18, 19, 20])
        elif model_name == 'pivoting_10_15':
            return np.random.choice([-15, -14, -13, -12, -11, 11, 12, 13, 14, 15])
        elif model_name == 'pivoting_5_10':
            return np.random.choice([-10, -9, -8, -7, -6, 6, 7, 8, 9, 10])
        elif model_name == 'pivoting_0_5':
            return np.random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])

    def _get_obs(self):

        # Tool's angle
        obs = self.sim.data.get_joint_qpos("tool")
        tools_angle = np.arctan2(2 * (obs[3] * obs[6] + obs[4] * obs[5]), (1 - 2 * (obs[5] ** 2 + obs[6] ** 2)))
        tools_angle = 180 * tools_angle / np.pi

        # Gripper's angle
        grippers_angle = (-1) * (self.sim.data.get_joint_qpos("kuka_joint_6"))
        grippers_angle = 180 * grippers_angle / np.pi
        self.grippers_angle = grippers_angle

        # Angle error between tool and gripper
        self.tool2gripper_angle = tools_angle - grippers_angle
        self.tool2gripper2desired_angle = self.tool2gripper_angle - self.desired_angle

        # Gripper's velocity
        grippers_vel = self.sim.data.get_joint_qvel("kuka_joint_6")

        # Gripper's position
        grippers_pos = self.sim.data.get_joint_qpos("kuka_joint_6")

        # Tool's velocity relative to gripper
        tools_vel = self.sim.data.get_joint_qvel("tool")
        tools_vel = tools_vel[5]  # Tool's global velocity
        tools_vel = tools_vel - grippers_vel

        # Rotational Joint
        rotational_joint = self.sim.data.get_joint_qvel("kuka_joint_7")

        # Tools position x y z
        tools_x = obs[0]
        tools_y = obs[1]
        tools_z = obs[2]
        if tools_z < 1.1:
            drop = 1
        else:
            drop = 0

        # Upper and Lower Gripper distance
        grippers_dist = self.sim.data.get_joint_qpos("gripper_joint_upper")

        # obs = np.concatenate(
        #     [[self.tool2gripper_angle - self.desired_angle], [tools_vel], [drop], [grippers_dist], [tools_x], [tools_y],
        #       [rotational_joint], [grippers_vel], [grippers_pos]] ).ravel()

        obs = np.concatenate(
            [[self.tool2gripper_angle - self.desired_angle], [tools_vel], [drop],
             [grippers_vel], [grippers_pos]]).ravel()

        return obs

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent + 0.7
