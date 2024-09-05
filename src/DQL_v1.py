import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from collections import deque
import os
import time
import shutil

#variables for loading and saving Q-table for future use
pre_trained=False
save=True

prev_test=1
curr_test=1

num_episodes_train = 500
num_episodes_test = 4

stability=50

shutdown=False

def shutdown_pc():
        os.system("shutdown /s /t 5")  # /s spegne il computer, /t 1 indica un ritardo di 1 secondo

def write_config(curr_test, num_episodes_train, num_episodes_test, stability, epsilon, epsilon_decay, epsilon_min, learn_rate, disc_factor, rep_mem_size, batch_size, step_limit):
    with open(f"Config_p{curr_test}_dqn_v1.txt", "w") as file:
        file.write(f"Part {curr_test} of training and testing\n")
        file.write(f"Number of training episodes={num_episodes_train}\n")
        file.write(f"Number of test episodes={num_episodes_test}\n")
        file.write(f"Stability={stability}\n")
        file.write(f"Epsilon={epsilon}\n")
        file.write(f"Epsilon Decay={epsilon_decay}\n")
        file.write(f"Epsilon Min={epsilon_min}\n")
        file.write(f"Alpha={learn_rate}\n")
        file.write(f"Gamma={disc_factor}\n")
        file.write(f"Replay memory size={rep_mem_size}\n")
        file.write(f"Batch size={batch_size}\n")
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

def save_training_state(target_dqn, memory, filename=f'Training_state_p{curr_test}_dqn_v1.pth'):
    full_path = os.path.join(os.getcwd(), filename)
    state = {
        #'policy_dqn_state_dict': policy_dqn.state_dict(),
        'target_dqn_state_dict': target_dqn.state_dict(),
        'replay_memory': memory.memory
    }
    #print(f"state before saving{state['target_dqn_state_dict']}")
    torch.save(state, full_path)
    print(f"Dql saved to {full_path}")

def load_training_state_1(filename=f'DQL_results\Dqn_v1\p{prev_test}\Training_state_p{prev_test}_dqn_v1.pth'):
    full_path = os.path.join(os.getcwd(), filename)
    if os.path.isfile(full_path):
        state = torch.load(full_path)
        print(f"Dql loaded from {full_path}")
        return state
    else:
        print(f"No saved dql found at {full_path}")
        return None
    
def load_training_state_2(filename=f'Training_state_p{curr_test}_dqn_v1.pth'):
    full_path = os.path.join(os.getcwd(), filename)
    if os.path.isfile(full_path):
        state = torch.load(full_path)
        print(f"Dql loaded from {full_path}")
        return state
    else:
        print(f"No saved dql found at {full_path}")
        return None

def seed_everything(seed: int, env):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    env.unwrapped.seed(seed)
    return

#INITIALIZATION OF GPU USE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

#DQN
class DQN(nn.Module):
    def __init__(self, state_size, h1_nodes, h2_nodes, action_size):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(state_size, h1_nodes)   # first fully connected layer
        self.fc2 = nn.Linear(h1_nodes, h2_nodes)     # second fully connected layer
        #self.fc3 = nn.Linear(h2_nodes, h3_nodes)     # third fully connected layer
        #self.fc4 = nn.Linear(h3_nodes, h4_nodes)     # fourth fully connected layer
        self.out = nn.Linear(h2_nodes, action_size)  # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
        x = F.relu(self.fc2(x)) # Apply rectified linear unit (ReLU) activation
        #x = F.relu(self.fc3(x)) # Apply rectified linear unit (ReLU) activation
        #x = F.relu(self.fc4(x)) # Apply rectified linear unit (ReLU) activation
        x = self.out(x)         # Calculate output
        return x

#Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)       #transition = (state, action, new_state, reward, terminated)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# Pitfall Deep Q-Learning
class PitfallDQL():

    x=1/(num_episodes_train-stability)
    #print(x)
    #y=1/num_episodes_train
    #print(y)

    # Hyperparameters

    explor_rate=1                                       #epsilon
    epsilon_min=0.2
    a=epsilon_min/explor_rate
    #print(a)
    epsilon_decay=a**x                                  #(exponential decay) so that last 30 episodes is stable at 0.2
    #print(f"epsilon_decay={epsilon_decay}")

    learn_rate = 0.0001                                   # learning rate (alpha)
    """
    learn_min = 0.01
    b=learn_min/learn_rate
    #print(b)
    learn_decay=b**y                                    #(exponential decay)
    #print(f"learn_decay={learn_decay}") 
    """
    
    disc_factor = 0.99                                  # discount rate (gamma)
    """
    disc_min = 0.01
    c=disc_min/disc_factor
    #print(c)
    disc_decay=c**y                                     #(exponential decay)
    #print(f"disc_decay={disc_decay}")
    """
    
    replay_memory_size = 100000                        # size of replay memory
    batch_size = 32                                    # size of the training data set sampled from the replay memory

    alpha_nd=0.8                                        #alpha in non deterministic formula
    

    # Neural Network
    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error.
    optimizer = None                # NN Optimizer. Later.

    #from 0 to 17
    #FULL_ACTIONS = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 
    #               'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 
    #               'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']    

    """
    def optimize(self, mini_batch, target_dqn, mean_loss_episode, episode):
        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, done in mini_batch:

            # Convert state, action, new_state to tensors
            state = np.array(state)
            action = np.array(action)
            new_state = np.array(new_state)

            state = torch.tensor(state, dtype=torch.float32).to(device)
            action = torch.tensor(action, dtype=torch.int64).to(device)
            new_state = torch.tensor(new_state, dtype=torch.float32).to(device)

            # Compute the current Q value using the target network
            current_q = target_dqn(state)[action]
            print(f'q_value={current_q}')
            current_q_list.append(current_q)
            
            
            if done: 
                # When in a terminated state, target q value should be set to the reward.
                target_q = reward
            else:
                with torch.no_grad():
                    next_max_q = target_dqn(new_state).max()
                    #print(f'next_q_value={next_max_q}')
                    target_q = reward + self.disc_factor *next_max_q

            # Adjust the specific action to the target that was just calculated
            target_q_list.append(target_q)

        # Stack the Q values to compute the loss
        current_q_batch = torch.stack(current_q_list)
        target_q_batch = torch.stack(target_q_list)
        #print(f'current_q_values for batch={current_q_batch}')
        #print(f'target_q_values for batch={target_q_batch}')

        # Compute the loss between the current Q values and the target Q values
        loss = self.loss_fn(current_q_batch, target_q_batch)
        #print(f'loss for batch={loss}')

        #print(f'mean_loss_episode before update: {mean_loss_episode[episode]}')
        #sum the loss for this step in the total loss sum (at the end of the episode to be divided for step_counter)
        mean_loss_episode[episode]+=loss
        #print(f'mean_loss_episode after update: {mean_loss_episode[episode]}')

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    """
    def optimize(self, mini_batch, target_dqn, mean_loss_episode, episode):
        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, done in mini_batch:
            # Convert state, action, new_state to tensors
            state = torch.tensor(state, dtype=torch.float32).to(device)
            action = torch.tensor(action, dtype=torch.int64).to(device)
            new_state = torch.tensor(new_state, dtype=torch.float32).to(device)
            reward = torch.tensor(reward, dtype=torch.float32).to(device)
            #done = torch.tensor(action, dtype=torch.int64).to(device)


            # Compute the current Q value using the target network
            #current_q = target_dqn(state).gather(0, action.unsqueeze(0))
            current_q = target_dqn(state)[action]
            current_q_list.append(current_q)
            #print(f'q_value={current_q}')

            if done:
                target_q = reward
            else:
                with torch.no_grad():
                    next_max_q = target_dqn(new_state).max()
                    #target_q = reward + self.disc_factor * next_max_q
                    target_q = (1 - self.alpha_nd) * current_q + self.alpha_nd * (reward + self.disc_factor * next_max_q)

            # Append target Q to the list
            target_q_list.append(target_q.squeeze(0))

        # Stack the Q values to compute the loss
        current_q_batch = torch.stack(current_q_list)#.squeeze()
        target_q_batch = torch.stack(target_q_list)
        #print(f'current_q_values for batch={current_q_batch}')
        #print(f'target_q_values for batch={target_q_batch}')

        # Compute the loss between the current Q values and the target Q values
        loss = self.loss_fn(current_q_batch, target_q_batch)
        #print(f'loss for batch={loss}')
        #print(f'loss item for batch={loss.item()}')

        # Update the mean loss for this episode
        mean_loss_episode[episode] += loss.item()
        #print(f'mean_loss_episode after update: {mean_loss_episode[episode]}')

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #return loss.item()
    
    def train(self, env, episodes=num_episodes_train):

        print("INIZIO FASE TRAINING")
        print('\n')

        state_size = 128
        num_actions = env.action_space.n
        
        memory = ReplayMemory(self.replay_memory_size)

        # Create target network. Number of nodes in the hidden layer can be adjusted.
        target_dqn = DQN(state_size=state_size, h1_nodes=512, h2_nodes=128, action_size=num_actions).to(device)
        
        if pre_trained:
            loaded_state = load_training_state_1()
            if loaded_state is not None:
                target_dqn.load_state_dict(loaded_state['target_dqn_state_dict'])
                memory.memory = loaded_state['replay_memory']                           

        # Policy network optimizer. 
        self.optimizer = torch.optim.Adam(target_dqn.parameters(), lr=self.learn_rate)                

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        total_reward_per_episode = np.zeros(num_episodes_train)

        # List to keep track of epsilon, alpha and disc decay
        epsilon_history = []
        #alpha_history = []
        #disc_history = []

        #Used for resetting env and starting new episode when certain amount of steps has been executed
        step_limit=17000

        #counters for number of exploration and exploitation choices per episode
        explor_count=np.zeros(num_episodes_train)
        exploit_count=np.zeros(num_episodes_train)

        #mean loss per episode (sum of the batches losses divided by the number of steps)
        mean_loss_episodes=np.zeros(num_episodes_train)
        
        #time elapsed for each episode and cumulative time
        time_per_episode=np.zeros(num_episodes_train)
        total_time_execution=0

        write_config(curr_test, num_episodes_train, num_episodes_test, stability, self.explor_rate, self.epsilon_decay, self.epsilon_min, self.learn_rate, self.disc_factor, self.replay_memory_size, self.batch_size, step_limit)

        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            done=False
            step_count=0
            start_time=time.time()

            # Agent navigates map until it dies three times, collect all treasures or steps limit is achieved
            while(not done):

                # Select action based on epsilon-greedy
                if random.random() < self.explor_rate:
                    # select random action
                    explor_count[i]+=1
                    action=env.action_space.sample()
                    #print(action) 
                else:
                    # select best action
                    exploit_count[i]+=1            
                    with torch.no_grad():
                        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                        action = target_dqn(state_tensor).argmax().item()
                        #print(action)

                # Execute action
                new_state, reward, done, _, info = env.step(action)

                if info['lives']<=0:
                    message='You died, GAME OVER!'
                    #print('You died, GAME OVER!')

                # Save experience into memory
                memory.append((state, action, new_state, reward, done)) 

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count+=1

                # Keep track of the rewards collected per episode.
                total_reward_per_episode[i] += reward

                # Copy policy network to target network after a certain number of steps
                if step_count >= step_limit:
                    message='Step Limit achieved, TIMEOUT!'
                    #print('Step Limit achieved, GAME OVER!')
                    done=True
                
                if len(memory) >= self.batch_size:
                    mini_batch = memory.sample(self.batch_size)
                    self.optimize(mini_batch, target_dqn, mean_loss_episodes, i) 

            #last phase to compute the mean loss for the current episode
            #print(f'mean_loss_episode[i] before division: {mean_loss_episodes[i]}')
            mean_loss_episodes[i]/=step_count
            #print(f'mean_loss_episode[i] after division: {mean_loss_episodes[i]}')

            time_per_episode[i]=(time.time()-start_time)
            total_time_execution+=time_per_episode[i]

            # Print episode statistics
            print(f"EPISODE {i + 1}/{num_episodes_train} ({message}):")
            print(f"Time elapsed: {time_per_episode[i]:.0f} s,")
            print(f"Steps Taken: {step_count},")
            print(f"Epsilon: {self.explor_rate:.4f},")
            #print(f"Alpha: {self.learn_rate:.4f},")
            #print(f"Disc_factor: {self.disc_factor:.4f},")
            print(f"Mean Loss: {mean_loss_episodes[i]:.2f},")
            print(f"Total Reward: {total_reward_per_episode[i]:.2f}")
            print('-------------------------------------------------')
            

            # Decay epsilon, alpha and disc_factor
            self.explor_rate = max(self.epsilon_min, self.explor_rate*self.epsilon_decay)
            epsilon_history.append(self.explor_rate)

            """
            self.learn_rate = max(self.learn_min, self.learn_rate*self.learn_decay)
            alpha_history.append(self.learn_rate)
            """

            """
            self.disc_factor = max(self.disc_min, self.disc_factor*self.disc_decay)
            disc_history.append(self.disc_factor)
            """

        # Close environment
        env.close()

        # Save policy
        if save:
            save_training_state(target_dqn, memory)

        # window of moving average
        window_size = 50

        # Moving average
        #moving_avg_reward = np.convolve(total_reward_per_episode, np.ones(window_size) / window_size, mode='valid')
        #moving_avg_loss = np.convolve(mean_loss_episodes, np.ones(window_size) / window_size, mode='valid')
        
        moving_avg_reward = [
            np.mean(total_reward_per_episode[i:i + window_size])
            for i in range(0, len(total_reward_per_episode), window_size)
        ]
        moving_avg_loss = [
            np.mean(mean_loss_episodes[i:i + window_size])
            for i in range(0, len(mean_loss_episodes), window_size)
        ]

        # Definisci l'asse x per la media mobile usando il centro di ogni blocco
        x_moving_avg = np.arange(window_size, episodes + 1, window_size)

        #Plotting

        plt.figure()
        plt.plot(range(1, episodes + 1), total_reward_per_episode, label='Total Reward per Episode', color='green')
        plt.plot(x_moving_avg, moving_avg_reward, label='Moving Average', color='red')
        #plt.plot(range(window_size, episodes + 1), moving_avg_reward, label='Moving Average', color='red')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'DQL Training p{curr_test}: Total Reward per Episode')
        plt.savefig(f'Training_Reward_p{curr_test}_dqn_v1.png')
        

        plt.figure()
        plt.plot(range(1, episodes + 1), mean_loss_episodes, label='Mean Loss per Episode', color='green')
        plt.plot(x_moving_avg, moving_avg_loss, label='Moving Average', color='red')
        #plt.plot(range(window_size, episodes + 1), moving_avg_loss, label='Moving Average', color='red')
        plt.xlabel('Episode')
        plt.ylabel('Mean Loss')
        plt.title(f'DQL Training p{curr_test}: Mean Loss per Episode')
        plt.savefig(f'Mean_Loss_p{curr_test}_dqn_v1.png')
        
        """
        plt.figure()
        plt.plot(range(1, episodes + 1), total_reward_per_episode)
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title(f'DQL Training p{curr_test}: Total Reward per Episode')
        plt.savefig(f'Training_Reward_p{curr_test}_dqn_v1.png')

        plt.figure()
        plt.plot(range(1, episodes + 1), mean_loss_episodes)
        plt.xlabel('Episodes')
        plt.ylabel('Mean Loss')
        plt.title(f'DQL Training p{curr_test}: Mean Loss per Episode')
        plt.savefig(f'Mean_Loss_p{curr_test}_dqn_v1.png')
        """

        plt.figure()
        plt.plot(range(1, episodes + 1), explor_count)
        plt.xlabel('Episodes')
        plt.ylabel('Explorations')
        plt.title(f'DQL Training p{curr_test}: Explorations per Episode')
        plt.savefig(f'Explorations_p{curr_test}_dqn_v1.png')

        plt.figure()
        plt.plot(range(1, episodes + 1), exploit_count)
        plt.xlabel('Episodes')
        plt.ylabel('Exploitations')
        plt.title(f'DQL Training p{curr_test}: Exploitations per Episode')
        plt.savefig(f'Exploitations_p{curr_test}_dqn_v1.png')

        plt.figure()
        plt.plot(range(1, episodes + 1), epsilon_history)
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        plt.title(f'DQL Training p{curr_test}: Epsilon value per Episode')
        plt.savefig(f'Epsilon_p{curr_test}_dqn_v1.png')

        """
        plt.figure()
        plt.plot(range(1, episodes + 1), alpha_history)
        plt.xlabel('Episodes')
        plt.ylabel('Alpha')
        plt.title(f'DQL Training p{curr_test}: Alpha value per Episode')
        plt.savefig(f'Alpha_p{curr_test}_dqn_v1.png')
        """

        """
        plt.figure()
        plt.plot(range(1, episodes + 1), disc_history)
        plt.xlabel('Episodes')
        plt.ylabel('Discount Factor')
        plt.title(f'DQL Training p{curr_test}: Disc_factor value per Episode')
        plt.savefig(f'Discount_p{curr_test}_dqn_v1.png')
        """

        #save time elapsed
        tot_time_minutes=total_time_execution/60
        mean_time_minutes=np.mean(time_per_episode)/60
        with open(f"Training_Time_p{curr_test}_dqn_v1.txt", "w") as file:
            if(tot_time_minutes<60):
                file.write(f"Total time elapsed={tot_time_minutes:.0f} min")
            else:
                file.write(f"Total time elapsed={(tot_time_minutes/60):.0f} h")
            file.write('\n')
            file.write(f"Mean time elapsed per episode={mean_time_minutes:.0f} min")

    def test(self, env, episodes=num_episodes_test):

        print("INIZIO FASE TESTING")
        print('\n')

        state_size = 128
        num_actions = env.action_space.n

        # Load learned policy
        target_dqn = DQN(state_size=state_size, h1_nodes=512, h2_nodes=128, action_size=num_actions).to(device)
        
        loaded_state = load_training_state_2()
        target_dqn.load_state_dict(loaded_state['target_dqn_state_dict'])
        target_dqn.eval()  # switch model to evaluation mode

        total_reward_per_episode=np.zeros(num_episodes_test) 
        steps_taken=np.zeros(num_episodes_test) 
        step_limit=17000

        #time elapsed for each episode and cumulative time
        time_per_episode=np.zeros(num_episodes_train)
        total_time_execution=0

        for i in range(episodes):
            state = env.reset()[0]
            done = False  
            step_count=0
            start_time=time.time()
            
            # Agent navigates game until it looses all lives, collects all treasures, or timeout.
            while not done:

                #select best action             
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                    action = target_dqn(state_tensor).argmax().item()
                    #print(action)


                # Execute action
                new_state, reward, done, _, info = env.step(action)

                if info['lives']<=0:
                    message='You died, GAME OVER!'
                    #print('You died... Game over')

                state=new_state

                step_count+=1
                
                total_reward_per_episode[i]+=reward

                # Copy policy network to target network after a certain number of steps
                if step_count >= step_limit:
                    message='Step Limit achieved, TIMEOUT!'
                    #print('Step Limit achieved... Game over!')
                    done=True

            steps_taken[i]=step_count

            time_per_episode[i]=(time.time()-start_time)
            total_time_execution+=time_per_episode[i]

            # Print episode statistics
            #print(f"Episode {i + 1}/{num_episodes_test}, Total Reward: {total_reward_per_episode[i]}, Steps Taken: {steps_taken[i]}")
            
            print(f"EPISODE {i + 1}/{num_episodes_test} ({message}):")
            print(f"Time elapsed: {time_per_episode[i]:.0f} s,")
            print(f"Total Reward: {total_reward_per_episode[i]:.2f},")
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

        x_moving_avg = np.arange(window_size, episodes + 1, window_size)

        #Plotting

        plt.figure()
        plt.plot(range(1, episodes + 1), total_reward_per_episode, label='Total Reward per Episode', color='green')
        plt.plot(x_moving_avg, moving_avg_reward, label='Moving Average', color='red')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'DQL Test p{curr_test}: Total Reward per Episode')
        plt.savefig(f'Test_Reward_p{curr_test}_dqn_v1.png')

        """
        plt.figure()
        plt.plot(range(1, num_episodes_test + 1), total_reward_per_episode)
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title(f'DQL Test p{curr_test}: Total Reward per Episode')
        plt.savefig(f'Test_Reward_p{curr_test}_dqn_v1.png')
        """

        plt.figure()
        plt.plot(range(1, num_episodes_test + 1), steps_taken)
        plt.xlabel('Episodes')
        plt.ylabel('Steps')
        plt.title(f'DQL Test p{curr_test}: Steps per Episode')
        plt.savefig(f'Test_Steps_p{curr_test}_dqn_v1.png')

        #save time elapsed
        tot_time_minutes=total_time_execution/60
        mean_time_minutes=np.mean(time_per_episode)/60
        with open(f"Test_Time_p{curr_test}_dqn_v1.txt", "w") as file:
            if(tot_time_minutes<60):
                file.write(f"Total time elapsed={tot_time_minutes:.0f} min")
            else:
                file.write(f"Total time elapsed={(tot_time_minutes/60):.0f} h")
            file.write('\n')
            file.write(f"Mean time elapsed per episode={mean_time_minutes:.0f} min")

if __name__ == '__main__':
    pitfall_dql = PitfallDQL()

    render_train=False
    env = gym.make('ALE/Pitfall-ram-v5', obs_type='ram', render_mode="human" if render_train else None)
    seed_everything(42, env)
    pitfall_dql.train(env=env)

    render_test=False
    env = gym.make('ALE/Pitfall-ram-v5', obs_type='ram', render_mode="human" if render_test else None)
    seed_everything(42, env)
    pitfall_dql.test(env=env)

    src = os.getcwd()
    dest = f'{src}\DQL_results\Dqn_v1\p{curr_test}'
    string = 'dqn_v1'

    move_file(src, dest, string)

    if(shutdown):
        shutdown_pc()