import numpy as np 
from scipy.special import softmax, logsumexp 

# get the machine epsilon
eps_ = 1e-10
max_ = 1e+10

# the replay buffer to store the memory 
class simpleBuffer:
    
    def __init__( self):
        self.lst = []
        
    def push( self, *args):
        self.lst = tuple([ x for x in args]) 
        
    def sample( self ):
        return self.lst

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Base agent class    %
%%%%%%%%%%%%%%%%%%%%%%%%%% 
'''

class baseAgent:
    
    def __init__( self, phi_dim, act_dim_train, act_dim_test):
        self.phi_dim       = phi_dim
        self.act_dim       = { 'train': act_dim_train,
                               'test':  act_dim_test}
        self.act_dim_train = act_dim_train
        self.act_dim_test  = act_dim_test
        self._init_psi()
        self._init_pi()
        self._init_critic()
        self._init_beliefs()
        self._init_memory()

    def _init_psi( self):
        self.Wpsi = 5 * np.eye( self.phi_dim)
        self.psi  = softmax( self.Wpsi, axis=1)

    def _init_pi( self):
        self.Wpi, self.pi, self.p_a1x = dict(), dict(), dict()
        self.Wpi['train'] = np.ones( [ self.phi_dim, self.act_dim_train]
                          ) * 1 / self.act_dim_train
                          
        self.Wpi['test']  = np.ones( [ self.phi_dim, self.act_dim_test]
                          ) * 1 / self.act_dim_test
        self.pi['train'] = softmax( self.Wpi['train'], axis=1)
        self.pi['test']  = softmax( self.Wpi['test'], axis=1)
        self.p_a1x['train'] = self.psi @ self.pi['train']
        self.p_a1x['test']  = self.psi @ self.pi['test']

    def _init_beliefs( self):
        '''Init the prior beliefs

        p_s(s): belief of the marginal state
        p1_a(a): belief of the action in the training stage
        p2_a(a): belief of the action in the testing stage
        '''
        self.p_x  = np.ones( [ self.phi_dim, 1]
                          ) * 1 / self.phi_dim
        self.p_s  = np.ones( [ self.phi_dim, 1]
                          ) * 1 / self.phi_dim
        self.p_a = dict()
        self.p_a['train'] = np.ones( [ self.act_dim_train, 1]
                          ) * 1 / self.act_dim_train
        self.p_a['test']  = np.ones( [ self.act_dim_test, 1]
                          ) * 1 / self.act_dim_test
        
    def _init_critic( self):
        self.q = dict()
        self.q['train'] = np.ones( [ self.phi_dim, self.act_dim_train]
                          ) * 1 / self.act_dim_train
        self.q['test']  = np.ones( [ self.phi_dim, self.act_dim_test]
                          ) * 1 / self.act_dim_test
    
    def _init_memory( self):
        self.memory = simpleBuffer()

    def plan_act( self, stage, stim):
        pass
        
    def get_act( self):
        return np.random.choice( self.act_space, p=self.p_a1x)
        
    def eval_act( self, act):
        return self.p_a1x[ act]
        
    def update( self):
        pass

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%
%      Naive agent       %
%%%%%%%%%%%%%%%%%%%%%%%%%% 
''' 
class IM_PsiPi_2( baseAgent):
    '''Update perception and action

    ψ： D[ ψ(x|s)|| \sum ]
    '''

    def __init__( self, obs_dim, action_dim, params=[]):
        super().__init__( obs_dim, action_dim)
        if len(params):
            self._load_hyper_params( params)

    def _load_hyper_params( self, params):
        self.lr_q   = params[0]
        self.lr_pi  = params[1]
        self.lr_a   = params[2]
        self.lr_psi = params[3]
        self.lr_x   = params[4]
        self.lr_t   = params[5]
        self.tau    = params[6]
        self.C      = params[7]
        self.cog_load = 0
    
    def plan_act(self, stage, stim):
        # xs x sa = xa 
        self.p_a1x = ( self.psi @ self.pi[stage])[ stim, :] 
        
    def update_q( self,):
        '''
        mnimize the prediction error
        min_q (rt - qt)^2
        where qt = Q(xt, at)
        '''
        # δ = rt - Q(xt, at)
        rpe = self.rew - self.q[ self.stage][ self.stim, self.act]
        self.q[ self.stage][ self.stim, self.act] += self.lr_q * rpe 

    def update_pi( self):
        '''
        minimize the KLD and the value
            min_π KL[ π||π*]
        where pi* = 1/τ * Q(s,a) + log p_a(a)
        Q(s,a) = ∑_x p(x|s)Q(x,a) 
        '''
        beta = np.clip( 1/eval(f'self.tau_{self.stage}'), eps_, max_)
        p_x1s = (self.p_x * self.psi).T / self.p_s  # sx
        Q_bel =  p_x1s @ self.q[ self.stage]  # sx x xa = sa 
        log_pi_target = beta * Q_bel + np.log( self.p_a[ self.stage].T + eps_)
        pi_target = np.exp( log_pi_target - logsumexp( 
                            log_pi_target, keepdims=True, axis=1))
        # gradient descent
        Wpi_grad = np.zeros_like( self.Wpi[ self.stage])
        for s in range( self.phi_dim):
            I_s = np.zeros( [ self.phi_dim, 1])
            I_s[ s, 0] = 1  
            pred = self.pi[ self.stage][[ s], :]
            target = pi_target[[ s], :]
            dL_dpi = 1 + np.log( pred + eps_) - np.log( target + eps_) #1a
            dpi_dw = pred * np.eye( self.act_dim[ self.stage]) - pred.T @ pred #aa 
            Wpi_grad += ( I_s * self.p_s) @ ( dpi_dw @ dL_dpi.T).T #x1 x 1a = xa 
        self.Wpi[ self.stage] -= self.lr_pi * Wpi_grad
        self.pi[ self.stage]  = np.exp( self.Wpi[ self.stage] 
                           - logsumexp( self.Wpi[ self.stage], keepdims=True, axis=1))

    def update_pa( self):
        target = (self.p_s.T @ self.pi).T # (1x @ xa).T = a1
        self.p_a += self.lr_a * ( target - self.p_a)

    def update_psi( self):
        '''
        minimize the KLD and the value
            min_π KL[ ψ||ψ*]
        where ψ* = 1/τ * ∑_a π(a|s)Q(x,a) + log p(s) - ∑_a π(a|s) log π(a|s)
        
        '''
        pi = self.pi[ self.stage]
        q  = self.q[ self.stage]
        beta = np.clip( 1/eval(f'self.tau_{self.stage}'), eps_, max_)
        Hpi  = - np.sum( pi * np.log( pi + eps_)
                        , keepdims=True, axis=1) # x1 
        log_psi_target = beta * self.q @ self.pi.T \
                        + np.log( self.p_s.T + eps_)\
                        + Hpi.T  # sa @ ax + 1x + 1x = sx  
        psi_target = np.exp( log_psi_target - logsumexp( 
                             log_psi_target, keepdims=True, axis=1))
        # gradient descent
        Wpsi_grad = np.zeros_like( self.Wpsi)
        for x in range( self.phi_dim):
            I_x = np.zeros( [ self.phi_dim, 1])
            I_x[ x, 0] = 1  
            pred = self.psi[[x], :]
            target = psi_target[[x], :]
            dL_dpi = 1 + np.log( pred + eps_) - np.log( target + eps_) #1a
            dpi_dw = pred * np.eye(self.phi_dim) - pred.T @ pred #aa 
            Wpsi_grad += (I_x * self.p_x) @ (dpi_dw @ dL_dpi.T).T #x1 x 1a = xa 
        self.Wpsi -= self.lr_psi * Wpsi_grad
        self.psi = np.exp( self.Wpsi - logsumexp( self.Wpsi, keepdims=True, axis=1))
        
    def update_ps( self):
        target = (self.p_x.T @ self.psi).T # (1s @ sx).T = x1
        self.p_s += self.lr_x * ( target - self.p_s)

    def update( self):
        self.stage, self.stim, self.act, self.rew  = self.memory.sample() 
        self.update_q()
        self.update_pi()
        self.update_pa()
        self.update_psi()
        self.update_ps()
        #self.update_tau()


