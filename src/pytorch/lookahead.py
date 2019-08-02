from collections import defaultdict
import torch
from torch.optim import Optimizer

class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = 0.5
        self.state = defaultdict(dict)
        self.counter = 0

        # State initialization
        for group in self.optimizer.param_groups:
            for fast_param in group['params']:
                state = self.state[fast_param]
                state['slow_param'] = fast_param.data.clone()

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.counter = (self.counter + 1) % self.k
        if self.counter == 0:
            for group in self.optimizer.param_groups:
                for fast_param in group['params']:
                    state = self.state[fast_param]
                    slow_param = state['slow_param']
                    slow_param.add_(self.alpha, fast_param - slow_param)
                    fast_param.data.copy_(slow_param)
        return loss
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
    
    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)