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
            variable_ = Variable(
                operators[op] (left.value,right.value),
                ap_op,
                (left,right)
            )
        else:
            variable_ = Variable(
                operators[op] (left.value,right.value),
                operators[op] (left.grad_,right.grad_),
                (left,right)
            )
        return variable_
    return closure

class History(object):
    def __init__(self,variable,history=None):
        if history is None:
            self.history = [(variable,'var')]
        else:
            self.history = history

class Variable(History):
    def __init__(self,value,grad_=1.,obj=None,history=None):
        super(Variable,self).__init__(value,history)
        if isinstance(value,Variable):
            self.value = value.value 
            self.grad_ = value.grad_
            self.obj = value.obj
        else:
            self.value = value 
            self.grad_ = grad_
            self.obj = obj
    
    def __add__(self,other):
        other = other if isinstance(other,Variable) else Variable(other,grad_=0.,obj='const')
        variable_ = makeVarOperator(self,other,'+')()
        self.history.append((variable_,'+'))
        return Variable(value=variable_,history=self.history)
    
    def __radd__(self,other):
        other = other if isinstance(other,Variable) else Variable(other,grad_=0.,obj='const')
        variable_ = makeVarOperator(other,self,'+')()
        self.history.append((variable_,'+'))
        return Variable(value=variable_,history=self.history)      
    
    def __sub__(self,other):
        other = other if isinstance(other,Variable) else Variable(other,grad_=0.,obj='const')
        variable_ = makeVarOperator(self,other,'-')()
        self.history.append((variable_,'-'))
        return Variable(value=variable_,history=self.history)
    
    def __rsub__(self,other):
        other = other if isinstance(other,Variable) else Variable(other,grad_=0.,obj='const')
        variable_ = makeVarOperator(other,self,'-')()
        self.history.append((variable_,'-'))
        return Variable(value=variable_,history=self.history)

    def __mul__(self,other):
        other = other if isinstance(other,Variable) else Variable(other,grad_=0.,obj='const')
        ap_op = self.value * other.grad_ + self.grad_ * other.value
        variable_ = makeVarOperator(
            left=self,
            right=other,
            op='*',
            topological=True,
            ap_op=ap_op
        )()
        self.history.append((variable_,'*'))
        return Variable(value=variable_,history=self.history)

    def __rmul__(self,other):
        other = other if isinstance(other,Variable) else Variable(other,grad_=0.,obj='const')
        ap_op = other.value * self.grad_ + other.grad_ * self.value
        variable_ = makeVarOperator(
            left=other,
            right=self,
            op='*',
            topological=True,
            ap_op=ap_op
        )()
        self.history.append((variable_,'*'))
        return Variable(value=variable_,history=self.history)

    def __truediv__(self,other):
        other = other if isinstance(other,Variable) else Variable(other,grad_=0.,obj='const')
        ap_op = (self.grad_ * other.value - self.value * other.grad_) / (other.value * other.value)
        variable_ = makeVarOperator(
            left=self,
            right=other,
            op='/',
            topological=True,
            ap_op=ap_op
        )()
        self.history.append((variable_,'/'))
        return Variable(value=variable_,history=self.history)
    
    def __rtruediv__(self,other):
        other = other if isinstance(other,Variable) else Variable(other,grad_=0.,obj='const')
        ap_op = (other.grad_ * self.value - other.value * self.grad_) / (self.value * self.value)
        variable_ = makeVarOperator(
            left=other,
            right=self,
            op='/',
            topological=True,
            ap_op=ap_op
        )()
        self.history.append((variable_,'/'))
        return Variable(value=variable_,history=self.history)

    def __pow__(self,n):
        variable_ = Variable(
            self.value ** n,
            0.0 if n == 0.0 else n * self.value ** (n - 1) * self.grad_,
            (self.obj,self)
        )
        self.history.append((variable_,'**'))
        return Variable(value=variable_,history=self.history)
    
    def linear(self):
        variable_ = Variable(
            self.value, 
            self.grad_,
            (self.obj,self)
        )
        self.history.append((variable_,'f_linear'))
        return Variable(value=variable_,history=self.history)
    
    def relu(self):
        variable_ = Variable(
            0.0 if self.value < 0.0 else self.value,
            0.0 if self.value < 0.0 else 1.0,
            (self.obj,self)
        )
        self.history.append((variable_,'f_relu'))
        return Variable(value=variable_,history=self.history)

    def sigmoid(self):
        s = lambda x: 1 / (1.0 + math.exp(-1.0*x))
        variable_ = Variable(
            s(self.value),
            s(self.value)*(1.0-s(self.value)),
            (self.obj,self)
        )
        self.history.append((variable_,'f_sigmoid'))
        return Variable(value=variable_,history=self.history)
    
    def sin(self):
        variable_ = Variable(
            math.sin(self.value),
            math.cos(self.value),
            (self.obj,self)
        )
        self.history.append((variable_,'f_sin'))
        return Variable(value=variable_,history=self.history)

    def exp(self):
        variable_ = Variable(
            math.exp(self.value),
            math.exp(self.value) * self.grad_,
            (self.obj,self)
        )
        self.history.append((variable_,'f_exp'))
        return Variable(value=variable_,history=self.history)
    
    def backward(self):
        grad = 1.0
        for x in reversed(self.history):
            grad *= x[0].grad_
        return grad

class Pico(Variable):  
    def __init__(self,value,grad=1.,name=''):
        super(Pico,self).__init__(
            Variable(value=value,grad_=grad,obj=name)
        )

def main():
    x = Pico(2.0,name='x')
    print((2*x).grad_)
    print((x*2).grad_)
main()