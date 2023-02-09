import math

from torch.optim.lr_scheduler import _LRScheduler

# Build Cyclical Learning Rate

"""
**Cyclical Learning Rate (CLR)**

One of the fastai library features is the cyclical learning rate scheduler. 
We can implement something similar inheriting the _LRScheduler class from 
the torch library. Following the original paper's pseudocode, 
this CLR Keras callback implementation, and making a couple of adjustments
to support cosine annealing with restarts, let's create our own CLR scheduler.

The implementation of this idea is quite simple. 
The base PyTorch scheduler class has the get_lr() method that 
is invoked each time when we call the step() method. 
The method should return a list of learning rates depending on the 
current training epoch. In our case, we have the same learning rate for all 
of the layers, and therefore, we return a list with a single value.

The next cell defines a CyclicLR class that expectes a single callback function. 
This function should accept the current training epoch and the base value of 
learning rate, and return a new learning rate value

"""

class CyclicLR(_LRScheduler):
    
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # base_lrs: https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]

# Not really sure what this function does so I just put it here
def triangular(step_size, max_lr, method='triangular', gamma=0.99):
    
    def scheduler(epoch, base_lr):
        period = 2 * step_size
        cycle = math.floor(1 + epoch/period)
        x = abs(epoch/step_size - 2*cycle + 1)
        delta = (max_lr - base_lr)*max(0, (1 - x))

        if method == 'triangular':
            pass  # we've already done
        elif method == 'triangular2':
            delta /= float(2 ** (cycle - 1))
        elif method == 'exp_range':
            delta *= (gamma**epoch)
        else:
            raise ValueError('unexpected method: %s' % method)
            
        return base_lr + delta
        
    return scheduler

# This needs to note. Dont know why
def cosine(t_max, eta_min=0):
    
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min)*(1 + math.cos(math.pi*t/t_max))/2
    
    return scheduler