import numpy as np

class NoamScheduler():
    '''
    Noam Learning Rate Scheduler.
    init params:
        optimizer       : an optimizer to adjust learning rate for
        init_lr         : An initial learning rate
        dim_model         : The output dimension of the model
        n_warmup_steps  : A number of warmup steps
    '''

    def __init__(self, optimizer, dim_model, n_warmup_steps):
        self._optimizer = optimizer
        self.dim_model = dim_model                  # The output dimension of the model
        self.n_warmup_steps = n_warmup_steps    # Number of warmup steps
        self.n_steps = 0


    def _get_lr_scale(self):
        '''
        Get learning rate scaling.

        dim_model^(-0.5) * min(n_steps^(-0.5), n_steps * n_warmup_steps^(-1.5))
        
        '''
        return (self.dim_model ** -0.5) * min(self.n_steps ** (-0.5),self.n_steps * self.n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self._get_lr_scale()
        return lr
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

