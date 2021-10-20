import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from scipy.optimize import minimize

class RLModel:
    '''Outer loop of the fit

    This class can instantiate a dynamic decision-making 
    model, given the agent name and the loss function.
    The instantiate can fit the chosen loss function
    '''
    
    def __init__( self, agent, mode='mle'):
        self.agent = agent 
        self.mode  = mode 
        
    def assign_data( self, data):
        self.train_data = data
        
    def likelihood_(self, data, params):
    
        neg_log_like = 0. 
        state_dim = len( data.state.unique())
        action_dim = len( range(3))
        agent= self.agent( state_dim, action_dim, params) 

        for i in range( data.shape[0]):

            # obtain si and ai
            state  = int(data.state[i])
            action = int(data.action[i])
            reward = data.reward[i]
            # store 
            agent.memory.push( state, action, reward, i)
            # evaluate action: get pi(ai|S = si)
            pi_state_action = agent.eval_action( state, action)
            # calculate neg_log_like 
            neg_log_like += - np.log( pi_state_action + 1e-20)
            # model update
            agent.update()

        return neg_log_like
    
    def mle_loss(self, params):
        '''Calculate total mle loss

        calculate the total mle loss through
        for loop. 
        '''
        sample_lst = self.train_data
        tot_nll = 0.
        for i in sample_lst:            
            data = self.train_data[i]
            tot_nll += self.likelihood_( data, params)        
        return tot_nll

    def fit( self, data, bnds, seed, init=[]):

        # prepar the list to store the fit results
        np.random.seed(seed)
        self.assign_data( data)
        num_params = len( bnds)

        if len(init) ==0:

            # init parameter 
            param0 = list()
            for i in range( num_params):
                # random init from the bounds 
                i0 = bnds[ i][0] + (bnds[ i][ 1] - bnds[ i][0]) * np.random.rand()
                param0.append( i0)
            print( f'init with params {param0}')

        else:
            # assign the parameter 
            param0 = init
            print( 'init with params: ', init)
        
        # start fit 
        res = minimize( self.mle_loss, param0, 
                            bounds= bnds, options={'disp': True})
        
        # store to the list
        print( f'''Fitted params: {res.x}, 
                    MLE loss: {res.fun}''')
        
        # select the optimal param set 
        param_opt = res.x
        loss_opt  = res.fun
        
        return param_opt, loss_opt
    
    def quick_mle( self, test_data, params):
        '''Calculate the predicted trajectories
        using fixed parameters
        '''
        test_set_ind = test_data.keys()
        tot_nll = 0.
        for i in test_set_ind:            
            data = test_data[i]
            tot_nll += self.likelihood_( data, params)        
        return tot_nll

    def predict( self, data=False, params=[]):
        '''Calculate the predicted trajectories
        using fixed parameters
        '''
        if data == False:
            data = self.train_data
        if len(params) == 0:
            params = self.param_opt
        
        # a blank df
        out_data = pd.DataFrame( columns=[ 'subject',
                                           'setSize', 
                                           'block',
                                           'state', 
                                           'action', 
                                           'reward', 
                                           'iter', 
                                           'correctAct', 
                                           'prob', 
                                           'accuracy', 
                                           'negLogLike', 
                                           'pi_complexity', 
                                           'psi_complexity',
                                           'pi_weights',
                                           'psi_weights',
                                           'tradeoff',
                                           'cogload'])
        sample_list = data.keys()

        # each sample contains human respose within a block 
        for i in sample_list:
            input_data = data[i].copy()
            out_sample = self.simulate( input_data, params)
            out_data = pd.concat( [out_data, out_sample], axis=0, sort=True)
        return out_data

    def simulate( self, data, params):
    
        state_dim = len( data.state.unique())
        action_dim = 3
        agent= self.agent( state_dim, action_dim, params) 
        data['prob']           = float('nan')
        data['negLogLike']     = float('nan')
        data['pi_complexity']  = float('nan')
        data['psi_complexity'] = float('nan')
        data['pi_weights']     = float('nan')
        data['psi_weights']    = float('nan')
        data['tradeoff']       = float('nan')
        data['cogload']        = float('nan')

        for i in range( data.shape[0]):

            # obtain st, at, and rt
            state = int(data.state[i])
            correct_act = int(data.correctAct[i])
            human_action = int(data.action[i])
            action = agent.get_action( state)
            reward = np.sum(action  == correct_act)
            
            # evaluate action: get p(ai|S = si)
            # for model with representations: p(ai|si) = \sum_x pi(ai|x)q(x|si)
            pi_state_action = agent.eval_action( state, correct_act)
            likelihood      = agent.eval_action( state, human_action)

            # record some vals
            data['action'][i]         = action
            data['reward'][i]         = reward
            data['prob'][i]           = pi_state_action
            data['negLogLike'][i]     = - np.log( likelihood + 1e-18)
            try:
                data['pi_complexity'][i]  = agent.pi_complexity()
            except:
                data['pi_complexity'][i]  = np.nan 
            try:
                data['psi_complexity'][i] = agent.psi_complexity()
            except:
                data['psi_complexity'][i] = np.nan 
            try:
                p_rep = ','.join([str(np.round(i,4)) for i in agent.psi[ state, :]])
                data.loc[ i, 'psi_weights']    = p_rep
            except:
                data.loc[ i, 'psi_weights']    = np.nan 
            try:
                p_pi  = ''
                for s in range(agent.pi.shape[0]): 
                    p = ','.join([str(np.round(i,4)) for i in agent.pi[ s, :]])
                    p_pi += '\\' * (s > 0) + p
                data.loc[ i, 'pi_weights']    = p_pi
            except:
                data.loc[ i, 'pi_weights']    = np.nan 
            try:
                data['tradeoff'][i]       = agent.tau
            except:
                try:
                    data['tradeoff'][i]       = 1/agent.beta
                except:
                    data['tradeoff'][i]       = np.nan 
            try:
                data['cogload'][i]        = agent.cog_load 
            except:
                data['cogload'][i]        = np.nan 

            # store 
            agent.memory.push( state, action, reward, i)
            
            # model update
            agent.update()            
            
        return data
        
        