import math
import numbers 

class Variable(object):
    def __init__(self,value,grad_=1,name=''):
        self.value = value 
        self.grad_ = grad_
        self.name = name

class Pico(object):
    def __init__(self,variable,history=None):
        self.variable = variable
        self.value = variable.value
        self.grad_ = variable.grad_
        if history is None:
            self.history = [(variable,'variable')]
        else:
            self.history = history

    def grad(self):
        return self.grad_

    def __add__(self,other):
        if isinstance(other,numbers.Real):
            variable_ = Variable(
                self.value + other,
                self.grad_ + other
            )
        else:
            variable_ = Variable(
                self.value + other.value,
                self.grad_ + other.grad_
            )
        self.history.append((variable_,'__add__'))
        return Pico(variable_,self.history)
        
    def __sub__(self,other):
        if isinstance(other,numbers.Real):
            variable_ = Variable(
                self.value - other,
                self.grad_ - other
            )
        else:
            variable_ = Variable(
                self.value - other.value,
                self.grad_ - other.grad_
            ) 
        self.history.append((variable_,'__sub__'))
        return Pico(variable_,self.history)
    
    def __mul__(self,other):
        if isinstance(other,numbers.Real):
            variable_ = Variable(
                self.value * other,
                self.grad_ * other
            )
        else:
            variable_ = Variable(
                self.value * other.value,
                self.value * other.grad_ + self.grad_ * other.value
            )
        self.history.append((variable_,'__mul__'))
        return Pico(variable_,self.history)

    def __div__(self,other):
        if isinstance(other,numbers.Real): 
            variable_ = Variable(
                self.value / other, ## ZeroDivisionError
                self.grad_ / other
            )
        else:
            variable_ =  Variable(
                self.value / other.value,
                (self.grad_ * other.value - self.value * other.grad_) / (other.value * other.value)
            )
        self.history.append((variable_,'__div__'))
        return Pico(variable_,self.history)

    def __pow__(self,n):
        variable_ = Variable(
            self.value ** n,
            0.0 if n == 0.0 else n * self.value ** (n - 1) * self.grad_
        )
        self.history.append((variable_,'__pow__'))
        return Pico(variable_,self.history)

    def linear(self):
        variable_ = Variable(
            self.value, 
            self.grad_
        )
        self.history.append((variable_,'f_linear'))
        return Pico(variable_,self.history)
    
    def relu(self):
        variable_ = Variable(
            0.0 if self.value < 0.0 else self.value,
            0.0 if self.value < 0.0 else 1.0
        )
        self.history.append((variable_,'f_relu'))
        return Pico(variable_,self.history)

    def sigmoid(self):
        s = lambda x: 1 / (1.0 + math.exp(-1.0*x))
        variable_ = Variable(
            s(self.value),
            s(self.value)*(1.0-s(self.value)),
        )
        self.history.append((variable_,'f_sigmoid'))
        return Pico(variable_,self.history)
    
    def __repr__(self):
        return "{0}".format(self.value) 
