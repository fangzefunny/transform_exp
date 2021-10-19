import numpy as np
import pandas as pd 

class task:
    '''A Transform experiemnt paradigm

    The experiment is splited into two stages:
        - train stage: a simple multiple arm bandit problem 
        - test stage: a simple cat problem using SHJ paradigm 
                      https://psycnet.apa.org/record/2011-17802-001
    '''

    def __init__( self, n_dim, n_val, args):
        self.n_dim = n_dim
        self.n_val = n_val
        self.args = args
        self._init_record()
    
    def _init_record( self):
        col_vars = [ 'block', 'stage', 'group', 'stim', 'act', 'rew']
        self.record = pd.DataFrame( columns=col_vars)

    def reset( self):
        self.task = dict()
        self.set_train( self.args.train_lst)
        self.set_test(  self.args.test_lst)

    def set_train( self, group, lst=False):
        if lst:
            # if there is a input list, just use it
            # to set up the trial 
            block_lst, stim_lst = lst 
        else:
            stim_lst = []
            block_lst = []
            # decide a random blocks
            blocks = range( self.n_dim)
            np.random.shuffle( blocks)
            for bi in blocks:
                # generate a mini-block  
                stims = [ self.n_val * bi + vi 
                          for vi in range( self.n_val)]
                # pseudo-random the mini-block
                for _ in range( self.args.n_rep):
                    block_lst += [ bi] * self.n_val 
                    np.random.shuffle( stims)
                    stim_lst  += stims 
        # add to the record
        self.record[ 'stim'] = stim_lst
        self.record['block'] = block_lst 
        self.record['stage'] = 'train'
        self.record['group'] = group

    def set_test( self, group, lst=False):
        if lst:
            pass 
        else:
            stim_lst = []
            block_lst = []
            # decide a random blocks
            blocks = range( self.n_dim)
            np.random.shuffle( blocks)
            for bi in blocks:
                # generate a mini-block  
                stims = [ self.n_val * bi + vi 
                          for vi in range( self.n_val)]
                # pseudo-random the mini-block
                for ri in range( self.args.n_rep):
                    block_lst += [ bi] * self.n_val 
                    np.random.shuffle( stims)
                    stim_lst  += stims 
            # add to the record
            self.record[ 'stim'] = stim_lst
            self.record['block'] = block_lst 
            self.record['stage'] = 'train'
            self.record['group'] = group
    
    def _train_trial( self, act):
        # get the state
        state = self.state_fn[ self.t] 
        # get reward
        rew = self.rew_fn( state, act)
        # move on
        self.t += 1
        return rew 

    def _train_block( self, b_type):
        # set the timeline to 0
        self.t = 0 
        self.state_fn = self.args[ b_type][ 'state']
        self.rew_fn   = self.args[ b_type][ 'rew_fn']


class con_bandit:
    '''Contextual Bandit Problem

    Input:
        n_cont: the number of contexts, 
        n_arms: the number of arms for each context
        n_rep:  repeat n times 

    Output:
        An instantiation of contextual m
    '''

    def __init__( self, n_dims, n_vals, n_arms, n_reps, mode,
                        cons_space=False, seed=2021):
        self.rng    = np.random.RandomState( seed)
        self.n_dims = n_dims
        self.n_vals = n_vals 
        self.n_arms = n_arms
        self.n_reps = n_reps
        self.mode   = mode 
        if cons_space:
            self.cons_space = cons_space
        else:
            self.cons_space = np.arange( 0, self.n_cons)
        self.arms_space = np.arange( 0, self.n_arms)

    def _init_record( self):
        col_vars = [ 'block', 'stage', 'group', 'stim', 'act', 'rew']
        self.record = pd.DataFrame( columns=col_vars)

    def reset( self, group, seed=2021):
        '''Reset the env 
            * choose a new random seed
            * set the done
            * restart the timeline, t=0 
            * generate a sequence of trials 
        '''
        self.rng = np.random.RandomState( seed)
        self.done = False
        self.t = 0
        self.stims = self._pesudo_random
        self._init_record()
        self._get_trials( group)
        self._get_rew_fn( group)

    def _get_rew_fn( self, group='cont', lst=False):
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
        # generate the correct answer to 
        # the stimuli
        if lst:
            pass
        else: 

            ## there is no input, we need
            #  to generate a set of reward function 
            if group == 'cont':   
                keys = { 0: 0, 1: 1, 2: 2,   # s
                         3: 2, 4: 3, 5: 0,   # c
                         6: 3, 7: 1, 8: 2 }  # t
            elif group == 'pos_trans':
                keys = { 0: 0, 1: 0, 2: 2,   # s0=s1
                         3: 2, 4: 3, 5: 3,   # c1=c2
                         6: 2, 7: 1, 8: 2 }  # t0=t3
            elif group == 'neg_trans':
                keys = { 0: 1, 1: 1, 2: 2,   # s
                         3: 0, 4: 3, 5: 0,   # c
                         6: 3, 7: 2, 8: 2 }  # t
        
        return lambda x, y: 1 if y == keys[x] else 0
        
    def _get_trials( self, group, lst=False):
        '''Generate a pesudo random experirmental set
        '''
        # get the stimuli
        if lst:
            pass 
        else:
            stim_lst = []
            block_lst = []
            # decide a random blocks
            blocks = range( self.n_dims)
            np.random.shuffle( blocks)
            for bi in blocks:
                # generate a mini-block  
                stims = [ self.n_vals * bi + vi 
                          for vi in range( self.n_vals)]
                # pseudo-random the mini-block
                for _ in range( self.n_reps):
                    block_lst += [ bi] * self.n_val 
                    np.random.shuffle( stims)
                    stim_lst  += stims 
            # add to the record
            self.record[ 'stim'] = stim_lst
            self.record['block'] = block_lst 
            self.record['stage'] = 'train'
            self.record['group'] = group
            self.train

    def step( self, act):
        # load the stimuli xt
        stim = self.record['stim'][self.t]
        # get the reward rt 
        rew  = self.rew_fn( stim, act)
        # make a step forward in the timeline 
        self.t += 1 
        # if this is the last trial
        if 
        