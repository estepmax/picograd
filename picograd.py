import math
import operator
import numbers 

class History(object):
    def __init__(self,variable,history=None):
        if history is None:
            self.history = [(variable,'variable')]
        else:
            self.history = history

class Variable(object):
    def __init__(self,value,grad_=1,name=''):
        self.value = value 
        self.grad_ = grad_
        self.name = name

def makeVarOperator(left,right,op,topological=False,ap_op=None):
    operators = {
        "+" : operator.add,
        "-" : operator.sub,
        "*" : operator.mul,
        "/" : operator.div
    }
    def closure():
        if isinstance(right,numbers.Real):
            variable_ = Variable(
                operators[op] (left.value,right),
                operators[op] (left.grad_,0.0),
                (left.vn,right)
            )
        else:
            if topological:
                variable_ = Variable(
                    operators[op] (left.value,right.value),
                    ap_op,
                    (left.vn,right.vn)
                )
            else:
                variable_ = Variable(
                    operators[op] (left.value,right.value),
                    operators[op] (left.grad_,right.grad_),
                    (left.vn,right.vn)
                )
        return variable_
    return closure

class Pico(History):
    def __init__(self,variable,history=None):
        super(Pico,self).__init__(variable,history)
        self.variable = variable 
        self.value = variable.value 
        self.grad_ = variable.grad_
        self.vn = variable.name

    def grad(self):
        return self.grad_

    def __add__(self,other):
        variable_ = makeVarOperator(self,other,'+')()
        self.history.append((variable_,'+'))
        return Pico(variable_,self.history)
        
    def __sub__(self,other):
        variable_ = makeVarOperator(self,other,'-')()
        self.history.append((variable_,'-'))
        return Pico(variable_,self.history)
    
    def __mul__(self,other):
        ap_op = self.value * other.grad_ + self.grad_ * other.value
        variable_ = makeVarOperator(
            left=self,
            right=other,
            op='*',
            topological=True,
            ap_op=ap_op
        )()
        self.history.append((variable_,'*'))
        return Pico(variable_,self.history)

    def __div__(self,other):
        ap_op = (self.grad_ * other.value - self.value * other.grad_) / (other.value * other.value)
        variable_ = makeVarOperator(
            left=self,
            right=other,
            op=None,
            topological=True,
            ap_op=ap_op
        )()
        self.history.append((variable_,'/'))
        return Pico(variable_,self.history)

    def __pow__(self,n):
        variable_ = Variable(
            self.value ** n,
            0.0 if n == 0.0 else n * self.value ** (n - 1) * self.grad_,
            (self.vn,)
        )
        self.history.append((variable_,'**'))
        return Pico(variable_,self.history)

    def linear(self):
        variable_ = Variable(
            self.value, 
            self.grad_,
            (self.vn,)
        )
        self.history.append((variable_,'f_linear'))
        return Pico(variable_,self.history)
    
    def relu(self):
        variable_ = Variable(
            0.0 if self.value < 0.0 else self.value,
            0.0 if self.value < 0.0 else 1.0,
            (self.vn,)
        )
        self.history.append((variable_,'f_relu'))
        return Pico(variable_,self.history)

    def sigmoid(self):
        s = lambda x: 1 / (1.0 + math.exp(-1.0*x))
        variable_ = Variable(
            s(self.value),
            s(self.value)*(1.0-s(self.value)),
            (self.vn,)
        )
        self.history.append((variable_,'f_sigmoid'))
        return Pico(variable_,self.history)
    
    def sin(self):
        variable_ = Variable(
            math.sin(self.value),
            math.cos(self.value),
            (self.vn,)
        )
        self.history.append((variable_,'f_cos'))
        return Pico(variable_,self.history)
        
    def __repr__(self):
        return "value: {0} \ngrad: {1}".format(self.value,self.grad_) 
    
    def backward(self):
        grad = 1.0
        for x in reversed(self.history):
            grad *= x[0].grad_
        return grad

def main():
    x = Variable(2.0)
    y = Pico(x)
    print(y.sigmoid())
main()