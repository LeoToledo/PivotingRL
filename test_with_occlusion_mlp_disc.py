import time
import gym
import pandas as pd
from keras.models import load_model
from PPO.PPO import PPO
import numpy as np
import pivoting_env
import yaml
from utils.utils import rescale_action_space, Occlusion
from utils.imputer import DataImputer
import IPython

# Read YAML file
with open('./parameters.yaml', 'r') as file_descriptor:
    parameters = yaml.load(file_descriptor)


def test(occlusion, imputer):
    global done
    print("============================================================================================")

    ################## hyperparameters ##################
    env_name = parameters['model']['env_name']
    has_continuous_action_space = True
    max_ep_len = parameters['model']['max_ep_len']  # max timesteps in one episode
    action_std = parameters['ppo']['action_parameters'][
        'min_action_std']  # set same std for action distribution which was used while saving

    render = parameters['test']['render']  # render environment on screen
    frame_delay = 0  # if required; add delay b/w frames

    total_test_episodes = parameters['test']['number_of_test_episodes']  # total num of testing episodes

    K_epochs = parameters['ppo']['hyperparameters']['k_epochs']  # update policy for K epochs
    eps_clip = parameters['ppo']['hyperparameters']['eps_clip']  # clip parameter for PPO
    gamma = parameters['ppo']['hyperparameters']['gamma']  # discount factor

    lr_actor = parameters['agent']['mlp']['lr_actor']  # learning rate for actor
    lr_critic = parameters['agent']['mlp']['lr_critic']  # learning rate for critic

    activate_occlusion = parameters['data_imputer']['activate_occlusion']
    failure_rate = parameters['model']['failure_rate']
    test_logs = parameters['test']['log_results']
    #####################################################

    env = gym.make(env_name)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = len(parameters['model']['ppo_acting_joints'])
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)

    # preTrained weights directory

    random_seed = 42  #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0  #### set this to load a particular checkpoint num

    directory = "PPO/PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + parameters['test']['model_name']
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    # Creates the dict to log steps
    if test_logs:
        logs_df = pd.DataFrame()

    # Loads the generator and discriminator models
    gen_model = load_model('utils/models/saved_models/simple_mlp_model.hdf5')
    disc_model = load_model('utils/models/saved_models/discriminator_local.hdf5')

    for ep in range(1, total_test_episodes + 1):

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Loads a different model each episode
        ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space,
                        action_std)
        directory = "PPO/PPO_preTrained" + '/' + env_name + '/'

        if parameters['test']['progressive']:
            # if ep <= 400:
            #     env.model_name = 'pivoting_25_30'
            # if ep <= 400:
            #     env.model_name = 'pivoting_20_25'
            # if ep <= 400:
            #     env.model_name = 'pivoting_15_20'
            # if ep <= 400:
            #     env.model_name = 'pivoting_10_15'
            # if ep <= 400:
            #     env.model_name = 'pivoting_5_10'
            if ep <= 400:
                env.model_name = 'pivoting_0_5'
        else:
            env.model_name = np.random.choice(['pivoting_25_30', 'pivoting_20_25', 'pivoting_15_20',
                                               'pivoting_10_15', 'pivoting_5_10', 'pivoting_0_5'])

        checkpoint_path = directory + env.model_name
        ppo_agent.load(checkpoint_path)
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len + 1):

            # Occlusion system
            if activate_occlusion:
                trigger_failure = True if (1 - np.random.random()) <= failure_rate else False
                if (occlusion.occlusion == True) or (occlusion.occlusion == False and trigger_failure):
                    occlusion.update_duration()

            # If the occlusion is up, insert ruid to tools angle
            if occlusion.occlusion:
                pos_neg = 1 if np.random.random() <= 0.5 else -1
                random_value = state[[0]] + pos_neg * 30 * np.random.random()
                state[[0]] = random_value

            # Data Imputer Prediction
            imputerX = []
            imputerX_dict = env.get_fail_obs
            for i in range(9):
                key = f"force_{i}"
                imputerX.append(imputerX_dict[key])
            imputerX = np.reshape(imputerX, (1, len(imputerX)))


            # Gen model
            synthetic_tool_angle = imputer.test_model(model=gen_model, testX=imputerX)

            # Disc model
            chosen_angle = choose_angle(imputerX=imputerX, generated=synthetic_tool_angle, real=state[[0]], model=disc_model)

            # Sets the chosen angle to be used
            state[[0]] = chosen_angle - env.grippers_angle - env.desired_angle

            #####################
            state[[3]] = state[[3]] * 100
            #####################

            action = ppo_agent.select_action(state)

            ####!!!!!@#####
            action = rescale_action_space(scale_factor=15, action=action)
            #######

            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        print(
            f"Episode : {ep} \t Timestep : {t} \t Average Reward : {int(ep_reward)} "
            f"\t Real : {int(env.get_current_angle())} "
            f"\t Target : {env.get_desired_angle()} \t Sucess : {done - env.get_drop_bool()}")

        # Append values to dataframe
        if test_logs:
            logs_dict = dict(episode=int(ep), duration=int(t), reward=int(ep_reward), final_angle=int(env.get_current_angle()),
                                target_angle=env.get_desired_angle())
            logs_df = logs_df.append(logs_dict, ignore_index=True)

    env.close()

    # Save log file into a csv
    if test_logs:
        logs_df.to_csv('results/results2/case4.csv', index=False)



def choose_angle(imputerX, generated, real, model):
    imputerX_gen = np.append(imputerX, generated)
    imputerX_real = np.append(imputerX, real)

    imputerX_gen = imputerX_gen.reshape(1, len(imputerX_gen))
    imputerX_real = imputerX_real.reshape(1, len(imputerX_real))

    angle_gen = model.predict(imputerX_gen)
    angle_real = model.predict(imputerX_real)

    chosen_angle = real if angle_real >= angle_gen else generated
    return chosen_angle


if __name__ == '__main__':
    occlusion = Occlusion()
    imputer = DataImputer()
    test(occlusion=occlusion, imputer=imputer)
