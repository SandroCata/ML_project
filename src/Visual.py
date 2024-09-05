import pickle
import torch
import os
import numpy as np

prev_test=1
version=1

def load_q_table(filename=f'TQL_results\Tql_v1\p{prev_test}\q_table_training_p{prev_test}_tql_v{version}.pkl'):
    full_path = os.path.join(os.getcwd(), filename)
    with open(full_path, 'rb') as f:
        q_table = pickle.load(f)
    print(f"Q-table loaded from {full_path}")
    return q_table

def load_training_state_1(filename=f'DQL_results_determ_formula\Dqn_v1\p{prev_test}\Training_state_p{prev_test}_dqn_v{version}.pth'):
    full_path = os.path.join(os.getcwd(), filename)
    if os.path.isfile(full_path):
        state = torch.load(full_path)
        print(f"Dql loaded from {full_path}")
        return state
    else:
        print(f"No saved dql found at {full_path}")
        return None

def print_dqn_info(dqn):
    print(f"DQN info p{prev_test}:")
    print(dqn['target_dqn_state_dict'])
    #print(dqn['replay_memory'])
    
if __name__ == '__main__':
   #Q = load_q_table2()
   #print(len(Q))
   DQN = load_training_state_1()
   #print_non_zero_q_values(Q)
   print_dqn_info(DQN)