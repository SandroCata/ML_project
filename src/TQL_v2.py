import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time
import shutil

#variables for loading and saving Q-table for future use
pre_trained=True
save=True

prev_test=5
curr_test=6

num_episodes_train = 500
num_episodes_test = 4

stability=50

shutdown=False

def shutdown_pc():
        os.system("shutdown /s /t 5")

def write_config(curr_test, num_episodes_train, num_episodes_test, stability, epsilon, epsilon_decay, epsilon_min, learn_rate, disc_factor, step_limit):
    with open(f"Config_p{curr_test}_tql_v2.txt", "w") as file:
        file.write(f"Part {curr_test} of training and testing\n")
        file.write(f"Number of training episodes={num_episodes_train}\n")
        file.write(f"Number of test episodes={num_episodes_test}\n")
        file.write(f"Stability={stability}\n")
        file.write(f"Epsilon={epsilon}\n")
        file.write(f"Epsilon Decay={epsilon_decay}\n")
        file.write(f"Epsilon Min={epsilon_min}\n")
        file.write(f"Alpha={learn_rate}\n")
        file.write(f"Gamma={disc_factor}\n")
        file.write(f"Step limit for environment reset={step_limit}\n")

def move_file(src, dest, string):

    if not os.path.exists(dest):
        os.makedirs(dest)
    
    files_to_move = [f for f in os.listdir(src) if string in f]
    
    for file in files_to_move:
        src_path = os.path.join(src, file)
        dest_path = os.path.join(dest, file)
        
        if os.path.isfile(src_path):
            shutil.move(src_path, dest_path)
            print(f"File '{file}' moved.")
        else:
            print(f"File '{file}' not found.")

def seed_everything(seed: int, env):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    env.unwrapped.seed(seed)
    return

def save_q_table(q_table, filename=f'q_table_training_p{curr_test}_tql_v2.pkl'):
    full_path = os.path.join(os.getcwd(), filename)
    with open(full_path, 'wb') as f:
        pickle.dump(q_table, f)
    print(f"Q-table saved to {full_path}")

def load_q_table(filename=f'TQL_results\Tql_v2\p{prev_test}\q_table_training_p{prev_test}_tql_v2.pkl'):
    full_path = os.path.join(os.getcwd(), filename)
    with open(full_path, 'rb') as f:
        q_table = pickle.load(f)
    print(f"Q-table loaded from {full_path}")
    return q_table

def load_q_table2(filename=f'q_table_training_p{curr_test}_tql_v2.pkl'):
    full_path = os.path.join(os.getcwd(), filename)
    with open(full_path, 'rb') as f:
        q_table = pickle.load(f)
    print(f"Q-table loaded from {full_path}")
    return q_table

def state_to_tuple(state):
    # Convert numpy array state to tuple (hashable)
    return tuple(state.tolist())

def train(env, pre_trained=pre_trained, save=save, num_episodes=num_episodes_train):

    print("INIZIO FASE TRAINING")
    print('\n')
    
    if pre_trained:
        #load the last trained Q-table
        Q = load_q_table()
        #print(Q)
    else:
        # Initialize Q-table as a dictionary
        Q = {}

    x=1/(num_episodes_train-stability)
    #print(x)
    #y=1/num_episodes_train
    #print(y)

    # Hyperparameters
    epsilon=1                                       #epsilon
    epsilon_min=0.2
    a=epsilon_min/epsilon
    #print(a)
    epsilon_decay=a**x                                  #(exponential decay) so that last 30 episodes is stable at 0.2
    #print(f"epsilon_decay={epsilon_decay}")

    alpha = 0.8                                   # learning rate (alpha)
    """
    learn_min = 0.01
    b=learn_min/learn_rate
    #print(b)
    learn_decay=b**y                                    #(exponential decay)
    #print(f"learn_decay={learn_decay}") 
    """
    
    gamma = 0.99                                  # discount rate (gamma) see how to modify during training
    """
    disc_min = 0.01
    c=disc_min/disc_factor
    #print(c)
    disc_decay=c**y                                     #(exponential decay)
    #print(f"disc_decay={disc_decay}")
    """

    #counters for number of exploration and exploitation choices per episode
    explor_count=np.zeros(num_episodes)
    exploit_count=np.zeros(num_episodes)

    # Training loop
    total_reward_per_episode=np.zeros(num_episodes)
    
    # List to keep track of epsilon, alpha and disc decay
    epsilon_history = []
    #alpha_history = []
    #disc_history = []

    step_limit=17000

    #time elapsed for each episode and cumulative time
    time_per_episode=np.zeros(num_episodes_train)
    total_time_execution=0

    write_config(curr_test, num_episodes_train, num_episodes_test, stability, epsilon, epsilon_decay, epsilon_min, alpha, gamma, step_limit)

    for episode in range(num_episodes):
        state, info = env.reset()
        current_lives=info['lives']
        #print(state)
        done = False
        step_count=0
        start_time=time.time()

        while not done:
            
            # Choose action: xxplore or exploit
            if np.random.rand() < epsilon:
                # Exploration: choose random action
                explor_count[episode]+=1
                action = env.action_space.sample()  
            else:
                # Exploitation: choose action with max Q-value for current state
                state_tuple = state_to_tuple(state)
                if state_tuple in Q:
                    exploit_count[episode]+=1
                    action_values = Q[state_tuple]
                    action = np.argmax(action_values)
                else:
                    # If state not in Q, explore
                    explor_count[episode]+=1
                    action =  env.action_space.sample()  

            next_state, reward, done, _, info = env.step(action)

            step_count+=1

            if step_count >= step_limit:
                message='Step Limit achieved, TIMEOUT!'
                done=True

            if info['lives']<=0:
                message='You died, GAME OVER!'

            #Positive reward if the action executed brings the agent to a new_state 
            #in which it has not lost lives or points
            if(reward==0):
                if(current_lives==info['lives']):
                    reward+=0.2
                else:
                    current_lives=info['lives']
                    reward-=200
            
            total_reward_per_episode[episode] += reward


            # Update Q-table:

            # Convert state to tuple
            state_tuple = state_to_tuple(state)
            next_state_tuple = state_to_tuple(next_state)

            # Update Q-value for (state, action) pair
            if state_tuple not in Q:
                # Initialize Q-values for new state
                Q[state_tuple] = np.zeros(env.action_space.n)  
            
            if next_state_tuple not in Q:
                # Initialize Q-values for new next state
                Q[next_state_tuple] = np.zeros(env.action_space.n)  

            # Q-learning update rule
            current_Q = Q[state_tuple][action]
            max_next_Q = np.max(Q[next_state_tuple])
            new_Q = new_Q = (1 - alpha) * current_Q + alpha * (reward + gamma * max_next_Q)
            Q[state_tuple][action] = new_Q

            # Update state
            state = next_state

        time_per_episode[episode]=(time.time()-start_time)
        total_time_execution+=time_per_episode[episode]
            

        # Print episode statistics
        print(f"EPISODE {episode + 1}/{num_episodes_train} ({message}):")
        print(f"Time elapsed: {time_per_episode[episode]:.0f} s,")
        print(f"Steps Taken: {step_count},")
        print(f"Epsilon: {epsilon:.4f},")
        #print(f"Alpha: {alpha:.4f},")
        #print(f"Disc_factor: {gamma:.4f},")
        print(f"Total Reward: {total_reward_per_episode[episode]:.2f}")
        print('-------------------------------------------------')

        # Decay epsilon, alpha and disc_factor
        epsilon = max(epsilon_min, epsilon*epsilon_decay)
        epsilon_history.append(epsilon)

        """
        alpha = max(learn_min, alpha*learn_decay)
        alpha_history.append(alpha)

        gamma = max(disc_min, gamma*disc_decay)
        disc_history.append(gamma
        """

    env.close()

    if save:
        save_q_table(Q)

    # window of moving average
    window_size = 50

    # Moving average
    #moving_avg_reward = np.convolve(total_reward_per_episode, np.ones(window_size) / window_size, mode='valid')
    moving_avg_reward = [
        np.mean(total_reward_per_episode[i:i + window_size])
        for i in range(0, len(total_reward_per_episode), window_size)
    ]

    # Definisci l'asse x per la media mobile usando il centro di ogni blocco
    x_moving_avg = np.arange(window_size, num_episodes + 1, window_size)

    #Plotting

    plt.figure()
    plt.plot(total_reward_per_episode, label='Total Reward per Episode')
    plt.plot(x_moving_avg, moving_avg_reward, label='Moving Average', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'TQL Training p{curr_test}: Total Reward per Episode')
    plt.savefig(f'Training_Reward_p{curr_test}_tql_v2.png')

    """
    plt.figure()
    plt.plot(range(1, num_episodes + 1), total_reward_per_episode)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title(f'TQL Training p{curr_test}: Total Reward per Episode')
    plt.savefig(f'Training_Reward_p{curr_test}_tql_v2.png')
    """
    
    plt.figure()
    plt.plot(range(1, num_episodes + 1), explor_count)
    plt.xlabel('Episodes')
    plt.ylabel('Explorations')
    plt.title(f'TQL Training p{curr_test}: Explorations per Episode')
    plt.savefig(f'Explorations_p{curr_test}_tql_v2.png')

    plt.figure()
    plt.plot(range(1, num_episodes + 1), exploit_count)
    plt.xlabel('Episodes')
    plt.ylabel('Exploitations')
    plt.title(f'TQL Training p{curr_test}: Exploitations per Episode')
    plt.savefig(f'Exploitations_p{curr_test}_tql_v2.png')

    plt.figure()
    plt.plot(range(1, num_episodes + 1), epsilon_history)
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title(f'TQL Training p{curr_test}: Epsilon value per Episode')
    plt.savefig(f'Epsilon_p{curr_test}_tql_v2.png')

    """
    plt.figure()
    plt.plot(range(1, num_episodes + 1), alpha_history)
    plt.xlabel('Episodes')
    plt.ylabel('Alpha')
    plt.title(f'TQL Training p{curr_test}: Alpha value per Episode')
    plt.savefig(f'Alpha_p{curr_test}_tql_v2.png')

    plt.figure()
    plt.plot(range(1, num_episodes + 1), disc_history)
    plt.xlabel('Episodes')
    plt.ylabel('Discount Factor')
    plt.title(f'TQL Training p{curr_test}: Disc_factor value per Episode')
    plt.savefig(f'Discount_p{curr_test}_tql_v2.png')
    """

    #save time elapsed
    tot_time_minutes=total_time_execution/60
    mean_time_seconds=np.mean(time_per_episode)
    with open(f"Training_Time_p{curr_test}_tql_v2.txt", "w") as file:
        if(tot_time_minutes<60):
            file.write(f"Total time elapsed={tot_time_minutes:.0f} min")
        else:
            file.write(f"Total time elapsed={(tot_time_minutes/60):.0f} h")
        file.write('\n')
        file.write(f"Mean time elapsed per episode={mean_time_seconds:.0f} min")
    
def test(env, num_episodes=num_episodes_test):
    
    print("INIZIO FASE TESTING")
    print('\n')

    Q = load_q_table2()
    
    total_reward_per_episode=np.zeros(num_episodes)

    state_in_Q = np.zeros(num_episodes)
    state_not_in_Q = np.zeros(num_episodes)

    step_taken=np.zeros(num_episodes)
    step_limit=17000

    #time elapsed for each episode and cumulative time
    time_per_episode=np.zeros(num_episodes_train)
    total_time_execution=0

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        step_count=0
        start_time=time.time()

        while not done:
            
            # Choose action: 
            state_tuple = state_to_tuple(state)
            if state_tuple in Q:
                #Exploitation: choose action with max Q-value for current state
                action_values = Q[state_tuple]
                action = np.argmax(action_values)
                state_in_Q[episode]+=1
                #print(f'Sono in argmax e lo stato esiste in Q: {action}')
            else:
                #Exploration: since that state was not discovered yet 
                action = env.action_space.sample()
                state_not_in_Q[episode]+=1
                #print(f'Sono in random perchè lo stato non esiste in Q: {action}')

            next_state, reward, done, _, info = env.step(action)

            total_reward_per_episode[episode] += reward

            step_count+=1

            if step_count >= step_limit:
                message='Step Limit achieved, TIMEOUT!'
                #print('Step Limit achieved... Game over!')
                done=True
            
            if info['lives']<=0:
                message='You died, GAME OVER!'
                #print('You died... Game over')
            
            # Update state and statistics
            state = next_state

        step_taken[episode]=step_count

        time_per_episode[episode]=(time.time()-start_time)
        total_time_execution+=time_per_episode[episode]

        # Print episode statistics
        print(f"EPISODE {episode + 1}/{num_episodes_test} ({message}):")
        print(f"Time elapsed: {time_per_episode[episode]:.0f} s,")
        print(f"Total Reward: {total_reward_per_episode[episode]:.2f},")
        print(f"Steps Taken: {step_count}")
        print('-------------------------------------------------')

    env.close()

    # window of moving average
    window_size = 1

    # Moving average
    #moving_avg_reward = np.convolve(total_reward_per_episode, np.ones(window_size) / window_size, mode='valid')
    moving_avg_reward = [
        np.mean(total_reward_per_episode[i:i + window_size])
        for i in range(0, len(total_reward_per_episode), window_size)
    ]

    # Definisci l'asse x per la media mobile usando il centro di ogni blocco
    x_moving_avg = np.arange(window_size, num_episodes + 1, window_size)

    #Plotting

    plt.figure()
    plt.plot(total_reward_per_episode, label='Total Reward per Episode')
    plt.plot(x_moving_avg, moving_avg_reward, label='Moving Average', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'TQL Test p{curr_test}: Total Reward per Episode')
    plt.savefig(f'Test_Reward_p{curr_test}_tql_v2.png')

    """
    plt.figure()
    plt.plot(range(1, num_episodes + 1), total_reward_per_episode)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title(f'TQL Test p{curr_test}: Total Reward per Episode')
    plt.savefig(f'Test_Reward_p{curr_test}_tql_v2.png')
    """

    plt.figure()
    plt.plot(range(1, num_episodes + 1), state_in_Q)
    plt.xlabel('Episodes')
    plt.ylabel('State in Q-table')
    plt.title(f'TQL Test p{curr_test}: Known states visited per Episode')
    plt.savefig(f'Test_States_in_Q_p{curr_test}_tql_v2.png')

    plt.figure()
    plt.plot(range(1, num_episodes + 1), state_not_in_Q)
    plt.xlabel('Episodes')
    plt.ylabel('State not in Q-table')
    plt.title(f'TQL Test p{curr_test}: Unknown states visited per Episode')
    plt.savefig(f'Test_states_not_in_Q_p{curr_test}_tql_v2.png')

    plt.figure()
    plt.plot(range(1, num_episodes_test + 1), step_taken)
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.title(f'TQL Test {curr_test}: Steps per Episode')
    plt.savefig(f'Test_Steps_p{curr_test}_tql_v2.png')

    #save time elapsed
    tot_time_minutes=total_time_execution/60
    mean_time_seconds=np.mean(time_per_episode)
    with open(f"Test_Time_p{curr_test}_tql_v2.txt", "w") as file:
        if(tot_time_minutes<60):
            file.write(f"Total time elapsed={tot_time_minutes:.0f} min")
        else:
            file.write(f"Total time elapsed={(tot_time_minutes/60):.0f} h")
        file.write('\n')
        file.write(f"Mean time elapsed per episode={mean_time_seconds:.0f} min")

if __name__ == '__main__':

    render_train=False
    env = gym.make('ALE/Pitfall-ram-v5', obs_type='ram', render_mode="human" if render_train else None)
    seed_everything(42, env)
    train(env=env)

    render_test=False
    env = gym.make('ALE/Pitfall-ram-v5', obs_type='ram', render_mode="human" if render_test else None)
    seed_everything(42, env)
    test(env=env)

    src = os.getcwd()
    dest = f'{src}\TQL_results\Tql_v2\p{curr_test}'
    string = 'tql_v2'

    move_file(src, dest, string)

    if(shutdown):
        shutdown_pc()


