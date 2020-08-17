import random
from picograd import Pico 

class Module(object):
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
    
    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self,n_in,activation=None):
        self.w =  [Pico(random.uniform(-1,1)) for _ in range(n_in)]
        self.b = Pico(0.0)
        self.activation = lambda x: x.relu() if activation is None else x.activation() ## tentative solution
    
    def __call__(self,value):
        x = sum((wk*vk for wk,vk in zip(self.w,value)),self.b)
        return self.activation(x)
    
    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
    def __init__(self,n_in,n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]
    
    def __call__(self,value):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out 
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

## Add MLP and ability to choose unique activations per layer 





    