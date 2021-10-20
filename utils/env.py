import numpy as np
import pandas as pd 
pd.options.mode.chained_assignment = None

def gen_stim( lst, r_dim, n_vals):
    new_lst = []
    r_dim -= 1

    if r_dim < 0:
        return lst 
    else:
        for i in lst:
            for j in range(n_vals):
                new_lst.append( i + str(j))
        
        return gen_stim( new_lst, r_dim, n_vals)

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

    def __init__( self, config, seed=1234):
    
                # group, n_dim=3, n_val=3, n_rep=2,
                #         seed=1234, config=False):
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
        self.group = config.group              
        self.n_dim = config.n_dim
        self.n_val = config.n_val
        self.n_rep_train = config.n_rep_train
        self.n_rep_test  = config.n_rep_test
        self.n_samp_test = config.n_samp_test 
        self.train_act_dim = 4
        self.test_act_dim  = 2
        self.config = config
        self.T = self.n_dim * self.n_val * self.n_rep_train +\
                 self.n_samp_test * self.n_rep_test
        self.reset()

    def reset( self):
        '''Init a environment 
        '''
        self.rew_fn = {}
        self.t = 0
        self.stage = 0
        # generate the experiment timeline 
        #self.timeline = self._get_train_blocks()
        self.timeline = pd.concat( [ self._get_train_blocks(),
                                        self._get_test_blocks()], ignore_index=True)
        # create the reward function 
        self.rew_fn['train'] = self._get_train_rew_fn()
        self.rew_fn['test'] = self._get_test_rew_fn()

    def feedback( self, act):
        '''Env's feedback to agent's action
        Input:
            action
        Output:
            reward, done 
        '''
        rew   = self.rew_fn[self.stage]( self.stim, act)
        done = False 
        if self.t == self.T:
            done = True 
        return rew, done 
    
    def step( self): 
        '''Observation in the environment
        Output:
            stim, stage
        '''
        # get all the observations from the environment
        self.stim  = self.timeline['stim'][ self.t]
        self.stage = self.timeline['stage'][ self.t]
        # make a step forward in the timeline 
        self.t += 1 
        return self.stim, self.stage
             
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
            for _ in range( self.n_rep_train):
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
        '''The block in SHJ paradigm
        In the testing stage, the subjects have to 
        complete a classification task, follows the SHJ
        paradigm.

        Each trial, the participants are shown 1 stimulus
        and are required to decides which of the two classes
        the stimulus falls into. 

        Each block contains 10 different stimuli. Each appear 
        twice randomly, so there are 20 trials within a block.
        '''
        # c1: 0,1 c2:2

        # init an empty timeline for testing  
        col_vars = [ 'block', 'stage', 'group', 'stim', 'act', 'rew']
        test_line = pd.DataFrame( columns=col_vars)
        block_lst = []
        stim_lst  = []

        for bi in range( self.n_rep_test):
            block_lst += [bi] * self.n_samp_test
            stims = self.stim_pool()
            self.rng.shuffle( stims)
            stim_lst += stims

        test_line[ 'stim'] = stim_lst
        test_line['block'] = block_lst 
        test_line['stage'] = 'test'
        test_line['group'] = self.group

        return test_line

    def stim_pool( self):
        cls1_pool = []
        cls2_pool = []
        nc = int( self.n_samp_test / 2)
        pool = gen_stim( [''], self.n_dim, self.n_val)
        for i in pool:
            if int(i[0]) < 2:
                cls1_pool.append(i)
            else:
                cls2_pool.append(i) 
        sel_pool = [ cls1_pool[idx] for idx in self.rng.choice( range( len(cls1_pool)), nc)] +\
                   [ cls2_pool[idx] for idx in self.rng.choice( range( len(cls2_pool)), nc)] 
        return sel_pool 

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
        if self.group == 0:  
            keys = { 0: 0, 1: 0, 2: 2,   # s0=s1
                        3: 2, 4: 3, 5: 3,   # c1=c2
                        6: 2, 7: 1, 8: 2 }  # t0=t3 
        elif self.group == 1:
            keys = { 0: 1, 1: 1, 2: 2,   # s
                        3: 0, 4: 3, 5: 0,   # c
                        6: 3, 7: 2, 8: 2 }  # t
        elif self.group == 2:
            keys = { 0: 0, 1: 1, 2: 2,   # s
                        3: 2, 4: 3, 5: 0,   # c
                        6: 3, 7: 1, 8: 2 }  # t
        
        return lambda x, y: 1 if y == keys[x] else 0

    def _get_test_rew_fn( self):
        '''Get reward function

           Notation:
           s: shape 
           t: texture
           c: color 

           For example: 

            Stim:  000  Info: s0, c0, t0
            Stim:  100  Info: s1, c0, t0
            Stim:  200  Info: s2, c0, t0
          
        '''
        return lambda x, y: 1 if y == (int(x[0]) > 1) else 0
    