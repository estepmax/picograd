import math
import operator
import numbers 

def makeVarOperator(left,right,op,topological=False,ap_op=None):
    operators = {
        "+" : operator.add,
        "-" : operator.sub,
        "*" : operator.mul,
        "/" : operator.truediv
    }
    def closure():  
        if topological:
            variable = Variable(
                operators[op] (left.value,right.value),
                ap_op,
                (left,right),
            )
        else:
            variable = Variable(
                operators[op] (left.value,right.value),
                operators[op] (left.grad,right.grad),
                (left,right)
            )
        return variable
    return closure

class History(object):
    def __init__(self,variable,history=None):
        if history is None:
            self.history = [(variable,'var')]
        else:
            self.history = history

class Variable(History):
    def __init__(self,value,grad=1.,obj=None,history=None):
        super(Variable,self).__init__(value,history)
        if isinstance(value,Variable):
            self.value = value.value 
            self.grad = value.grad
            self.obj = value.obj
        else:
            self.value = value 
            self.grad = grad
            self.obj = obj
    
    def __neg__(self):
        variable = Variable(
            -1 * self.value,
            -1 * self.grad,
            (self,'--'),
            self.history
        )
        variable.history.append((variable,'--'))
        return variable

    def __add__(self,other):
        other = other if isinstance(other,Variable) else Variable(other,grad=0.,obj='const')
        variable = makeVarOperator(self,other,'+')()
        variable.history = self.history
        variable.history.append((variable,'+'))
        return variable
    
    def __radd__(self,other):
        other = other if isinstance(other,Variable) else Variable(other,grad=0.,obj='const')
        variable = other + self
        return variable   
    
    def __sub__(self,other):
        other = other if isinstance(other,Variable) else Variable(other,grad=0.,obj='const')
        variable = makeVarOperator(self,other,'-')()
        variable.history = self.history 
        variable.history.append((variable,'-'))
        return variable
    
    def __rsub__(self,other):
        other = other if isinstance(other,Variable) else Variable(other,grad=0.,obj='const')
        variable = other - self
        return variable

    def __mul__(self,other):
        other = other if isinstance(other,Variable) else Variable(other,grad=0.,obj='const')
        ap_op = self.value * other.grad + self.grad * other.value
        variable = makeVarOperator(
            left=self,
            right=other,
            op='*',
            topological=True,
            ap_op=ap_op
        )()
        variable.history = self.history
        variable.history.append((variable,'*'))
        return variable

    def __rmul__(self,other):
        other = other if isinstance(other,Variable) else Variable(other,grad=0.,obj='const')
        variable = other * self
        return variable

    def __truediv__(self,other):
        other = other if isinstance(other,Variable) else Variable(other,grad=0.,obj='const')
        ap_op = (self.grad * other.value - self.value * other.grad) / (other.value * other.value)
        variable = makeVarOperator(
            left=self,
            right=other,
            op='/',
            topological=True,
            ap_op=ap_op
        )()
        variable.history = self.history
        variable.history.append((variable,'/'))
        return variable
    
    def __rtruediv__(self,other):
        other = other if isinstance(other,Variable) else Variable(other,grad=0.,obj='const')
        variable = other / self
        return variable

    def __pow__(self,n):
        variable = Variable(
            self.value ** n,
            0.0 if n == 0.0 else n * self.value ** (n - 1) * self.grad,
            (self.obj,self),
            self.history
        )
        variable.history.append((variable,'**'))
        return variable
    
    def linear(self):
        variable = Variable(
            self.value, 
            self.grad,
            (self.obj,self),
            self.history
        )
        variable.history.append((variable,'f_linear'))
        return variable
    
    def relu(self):
        variable = Variable(
            0.0 if self.value < 0.0 else self.value,
            0.0 if self.value < 0.0 else 1.0,
            (self.obj,self),
            self.history
        )
        variable.history.append((variable,'f_relu'))
        return variable

    def sigmoid(self):
        s = lambda x: 1 / (1.0 + math.exp(-1.0*x))
        variable = Variable(
            s(self.value),
            s(self.value)*(1.0-s(self.value)),
            (self.obj,self),
            self.history
        )
        variable.history.append((variable,'f_sigmoid'))
        return variable
    
    def sin(self):
        variable = Variable(
            math.sin(self.value),
            math.cos(self.value) * self.grad,
            (self.obj,self),
            self.history
        )
        variable.history.append((variable,'f_sin'))
        return variable

    def exp(self):
        variable = Variable(
            math.exp(self.value),
            math.exp(self.value) * self.grad,
            (self.obj,self),
            self.history
        )
        variable.history.append((variable,'f_exp'))
        return variable
    
    def backward(self):
        grad = 1.0
        for x in reversed(self.history):
            grad *= x[0].grad
        return grad

class Pico(Variable):  
    def __init__(self,value,grad=1.,name=''):
        super(Pico,self).__init__(
            Variable(value=value,grad=grad,obj=name)
        )
