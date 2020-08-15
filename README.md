# 10^-12
Picograd is a small automatic differentiation tool

## Usage

## Variable creation and registration

```python
x = Variable(2.0,name='x') ## creation
y = Pico(x) ## registration
```

## Plotting the derivative of sin(sin(x))

```python
import matplotlib.pyplot as plt
arange = [i*0.1 for i in range(-200,200)]
f = [] 
df = [] 
for k in arange:
    x = Variable(k)
    y = Pico(x)
    g = y.sin().sin()
    f.append(g.value)
    df.append(g.grad_)
plt.plot(f,label="sin(sin(x))")
plt.plot(df,label="d(sin(sin(x)))")
plt.legend()
plt.show()
```

![sin_sin_der](https://github.com/estepmax/picograd/blob/master/screenshots/sinsin.png)

