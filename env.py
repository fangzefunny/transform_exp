import numpy as np
import pandas as pd 

class task:
    '''A Transform experiemnt paradigm

    The experiment involves the following concepts:

    * Stage: the experiment is splited into two stages:
        - 0: Train stage. A simple multi-arm bandit problem
        - 1: Test stage. A simple classification problem using
             SHJ paradigm: https://psycnet.apa.org/record/2011-17802-001

    * Block: A stage inludes n_block blocks. The sequence of the block are pesudo
             randomed
    
    * Trial: A block incorporates n_trial trials. 
    
    * Group: to control the experiment setting
        - 0: Postive transfer. The training stage benefits the testing stage.
        - 1: Negative transfer. The training stage impairs the testing stage.
        - 2: Control transfer. The training stage has no impact on the testing stage.
    '''

    def __init__( self, group, n_dim=3, n_val=3, n_rep=2,
                        seed=1234, config=False):
        '''Init the environment
        args : the configuration of the experiment
        n_dim: the number of different dimensions. default 3
            - shape (s)
            - color (c) 
            - texture (t)
        n_val: the number of value in each dimension. default 3
            - s0: triangle
            - s1: square 
            - s3: cricle
        n_rep: present each stimuli repeatedly for n_rep times. 
        '''
        self.rng   = np.random.RandomState( seed)
        self.group = group
        self.n_dim = n_dim
        self.n_val = n_val
        self.n_rep = n_rep
        self.config = config
        self.T = self.n_dim * self.n_val * self.n_rep 
        self.reset()

    def reset( self):
        '''
        '''
        self.t = 0 
        if self.config:
            pass 
        else: 
            self.rew_fn = {}
            self.t = 0
            self.stage = 0
            # generate the experiment timeline 
            self.timeline = self._get_train_blocks()
            # self.timeline = pd.concat( [ self._get_train_blocks(),
            #                              self._get_test_blocks()])
            # create the reward function 
            self.rew_fn['train'] = self._get_train_rew_fn()
            #self._get_test_rew_fn()
    
    def step( self, act): 
        # get all the observations from the environment
        stim  = self.timeline['stim'][ self.t]
        stage = self.timeline['stage'][ self.t]
        rew   = self.rew_fn[stage]( stim, act)
        # make a step forward in the timeline 
        self.t += 1 
        done = False 
        if self.t == self.T:
            done = True 
        return stim, stage, rew, done 
             
    def _get_train_blocks( self,):

        # init an empty timeline for training 
        col_vars = [ 'block', 'stage', 'group', 'stim', 'act', 'rew']
        train_line = pd.DataFrame( columns=col_vars)
        stim_lst = []
        block_lst = []
        # decide a random blocks
        blocks = np.arange( self.n_dim)
        self.rng.shuffle( blocks)
        for bi in blocks:
            # generate a mini-block  
            stims = [ self.n_val * bi + vi 
                        for vi in range( self.n_val)]
            # pseudo-random the mini-block
            for _ in range( self.n_rep):
                block_lst += [ bi] * self.n_val 
                self.rng.shuffle( stims)
                stim_lst  += stims 
        # add to the record
        train_line[ 'stim'] = stim_lst
        train_line['block'] = block_lst 
        train_line['stage'] = 'train'
        train_line['group'] = self.group

        return train_line

    def _get_test_blocks( self,):
        return []

    def _get_train_rew_fn( self,):
        '''Get reward function

           Notation:
           s: shape 
           t: texture
           c: color 

           Stim:  0  Info: s1 
           Stim:  1  Info: s2
           Stim:  2  Info: s3
           Stim:  3  Info: c1
           Stim:  4  Info: c2
           Stim:  5  Info: c3
           Stim:  6  Info: t1
           Stim:  7  Info: t2
           Stim:  8  Info: t3
        '''
        ## there is no input, we need
        #  to generate a set of reward function 
        if group == 0:  
            keys = { 0: 0, 1: 0, 2: 2,   # s0=s1
                        3: 2, 4: 3, 5: 3,   # c1=c2
                        6: 2, 7: 1, 8: 2 }  # t0=t3 
        elif group == 1:
            keys = { 0: 1, 1: 1, 2: 2,   # s
                        3: 0, 4: 3, 5: 0,   # c
                        6: 3, 7: 2, 8: 2 }  # t
        elif group == 2:
            keys = { 0: 0, 1: 1, 2: 2,   # s
                        3: 2, 4: 3, 5: 0,   # c
                        6: 3, 7: 1, 8: 2 }  # t
        
        return lambda x, y: 1 if y == keys[x] else 0

    def _get_test_rew_fn( self, group):
        '''Get reward function

           Notation:
           s: shape 
           t: texture
           c: color 

           Stim:  0  Info: s1 
           Stim:  1  Info: s2
           Stim:  2  Info: s3
           Stim:  3  Info: c1
           Stim:  4  Info: c2
           Stim:  5  Info: c3
           Stim:  6  Info: t1
           Stim:  7  Info: t2
           Stim:  8  Info: t3
        '''
        ## there is no input, we need
        #  to generate a set of reward function 
        if group == 0:  
            keys = { 0: 0, 1: 0, 2: 2,   # s0=s1
                        3: 2, 4: 3, 5: 3,   # c1=c2
                        6: 2, 7: 1, 8: 2 }  # t0=t3 
        elif group == 1:
            keys = { 0: 1, 1: 1, 2: 2,   # s
                        3: 0, 4: 3, 5: 0,   # c
                        6: 3, 7: 2, 8: 2 }  # t
        elif group == 2:
            keys = { 0: 0, 1: 1, 2: 2,   # s
                        3: 2, 4: 3, 5: 0,   # c
                        6: 3, 7: 1, 8: 2 }  # t
        
        return lambda x, y: 1 if y == keys[x] else 0

if __name__ == '__main__':

    group = 0
    env = task( group)
    done = False
    t = 0 
    while not done:
        act = np.random.choice(4)
        stim, stage, rew, done = env.step( act)
        print( f'''t={t} 
                    stim: {stim}, stage: {stage}, reward: {rew}''')
        t += 1

    