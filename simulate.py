import numpy as np

# import the local packages 
from utils.env import task 

class task_config:

    def __init__( self,):
        self.group = 0              
        self.n_dim = 3
        self.n_val = 3
        self.n_rep_train = 2
        self.n_rep_test  = 2
        self.n_samp_test = 10

def sim_subj( env, agent,):
    '''Simulate the response of an agent

        -c: stage
        -x: observed stimuli
        -s: internal state 
        -a: action
        -r: reward 
        -π: policy
        -ψ: perception
    '''
    # init for the training 
    done = False
    train_history = [] 
    test_history  = []
    # start training 
    while not done:
        # see stimuli: 
        stim, stage = env.step()
        # choose action: a ~ π(a|s,c)
        act = agent.get_act( stage, stim)
        # get feedback: r = env(a)
        rew, done = env.feedback( act)
        # save for visualization: 
        eval(f'{stage}_history').append( rew)
        # the learning stage for the agent 
        agent.memory.push( stage, stim, act, rew)
        agent.update()

    return train_history, test_history

if __name__ == '__main__':

    config = task_config()
    env = task( config)
    
