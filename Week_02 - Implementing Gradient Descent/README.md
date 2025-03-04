# 🎯 Goals for week 02

1. See backpropagation in action by implementing a `Value` class and visualizing its computational graph.
2. Do manual backpropagation.
3. Implement automatic backpropagation.
4. Use a hyperbolic tangent as an activation function.
5. Practice writing high quality code:
   1. Easy to read.
   2. Safe from bugs.
   3. Ready for change.

## Task 1

**Description:**

Create a class `Value` that stores a single floating point number and implements the output operator.

**Test cases:**

```python
def main() -> None:
    value1 = Value(5)
    print(value1)

    value2 = Value(6)
    print(value2)
```

```console
Value(data=5)
Value(data=6)
```

## Task 2

**Description:**

Extend the `Value` class by implementing functionality to add two values.

**Test case:**

```python
def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    result = x + y
    print(result)
```

```console
Value(data=-1.0)
```

## Task 3

**Description:**

Extend the `Value` class by implementing functionality to multiply two values.

**Test case:**

```python
def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result)
```

```console
Value(data=4.0)
```

## Task 4

**Description:**

Extend the `Value` class with another state variable that holds the values that produced the current value.

**Test case:**

```python
def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result._prev)
```

```console
{Value(data=-6.0), Value(data=10.0)}
```

## Task 5

**Description:**

Extend the `Value` class with another state variable that holds the operation that produced the current value.

**Test case:**

```python
def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result._op)
```

```console
+
```

## Task 6

**Description:**

Implement a function that takes a `Value` object and returns the nodes and edges that lead to the passed object.

**Test case:**

```python
def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    
    nodes, edges = trace(x)
    print('x')
    print(f'{nodes=}')
    print(f'{edges=}')
    
    nodes, edges = trace(y)
    print('y')
    print(f'{nodes=}')
    print(f'{edges=}')
    
    nodes, edges = trace(z)
    print('z')
    print(f'{nodes=}')
    print(f'{edges=}')
    
    nodes, edges = trace(result)
    print('result')
    print(f'{nodes=}')
    print(f'{edges=}')
```

```console
x
nodes={Value(data=2.0)}
edges=set()
y
nodes={Value(data=-3.0)}
edges=set()
z
nodes={Value(data=10.0)}
edges=set()
result
nodes={Value(data=10.0), Value(data=-3.0), Value(data=4.0), Value(data=-6.0), Value(data=2.0)}
edges={(Value(data=-6.0), Value(data=4.0)), (Value(data=10.0), Value(data=4.0)), (Value(data=-3.0), Value(data=-6.0)), (Value(data=2.0), Value(data=-6.0))}
```

## Task 7

**Description:**

Let's visualize the directed acyclic graph (`DAG`) leading up to a certain value. We'll be using the Python package [graphviz](https://pypi.org/project/graphviz/). Note, that before installing it with `pip`, you should have `graphviz` installed on your operating system. Graphviz is available for installation [here](https://graphviz.org/download/). After installing, run the command `pip install graphiviz`.

Add the following code and ensure the test case runs successfully. Note that if by now you've run all scripts from the integrated terminal in vscode, you should now run this script from a terminal/command prompt that is **not** in VSCode.

```python
def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(filename='01_result', format='svg', graph_attr={
                           'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        dot.node(name=uid, label=f'{{ data: {n.data} }}', shape='record')
        if n._op:
            # if this value is a result of some operation, create an "op" node for the operation
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to the node of the operation
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the "op" node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
```

**Test case:**

```python
def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    
    # This will create a new directory and store the output file there.
    # With "view=True" it'll automatically display the saved file.
    draw_dot(result).render(directory='./graphviz_output', view=True)
```

![w02_01_result](../assets/w02_01_result.svg?raw=true "w02_01_result.png")

## Task 8

**Description:**

Include a label of the node in the visualization shown by `graphviz`.

**Test case:**

```console
python task08.py
```

![w02_02_result](../assets/w02_02_result.svg?raw=true "w02_02_result.png")

## Task 9

**Description:**

Now we're going to start back-propagating derivatives to see how slight nudges (changes) to each of the variables change the value of the final variable - `L`.

Extend the `Value` class by adding a new state variable that holds the gradient (derivative value) of that value with respect to `L`. By default it should be initialized with `0`, meaning "no effect" (initially we're assuming that the values do not affect the output).

Visualize the gradient of each node via `graphviz`.

**Test case:**

```console
python task09.py
```

![w02_03_result](../assets/w02_03_result.svg?raw=true "w02_03_result.png")

## Task 10

**Description:**

Manually do backpropagation on the generated computation graph.

![w02_03_result](../assets/w02_03_result.svg?raw=true "w02_03_result.png")

**Acceptance criteria:**

1. The process through which the values for the gradients are calculated is shown in comments.
2. A function `manual_der` is defined that helps verify the calculations.

**Test case:**

![w02_04_result](../assets/w02_04_result.svg?raw=true "w02_04_result.png")

## Task 11

**Description:**

Increase all values using the gradient with `0.01` to get closer to $0$. Print the new value of the loss function.

> **Note:** In the real world, you'll always have to minimize $L$ (so you'll have to subtract). However, here, because we're playing, this will not be case.

**Test case:**

```console
python task11.py
```

```console
Old L = -8.0
New L = -7.286496
```

## Task 12

**Description:**

Implement a perceptron with two inputs (for now without an activation function). Name the output node (on which you're calling `draw_dot`) `logit` - this is the term for a value that has not been passed through an activation function.

Here's what the perceptron model looks like:

![w02_neuron_model](../assets/w02_neuron_model.jpeg?raw=true "w02_neuron_model.jpeg")

You can see from the test case what configuration you have to use for `x1`, `x2`, `w1`, `w2` and `b`.

**Test case:**

```console
python task12.py
```

![w02_05_result](../assets/w02_05_result.svg?raw=true "w02_05_result.svg")

## Task 13

**Description:**

Add the hyperbolic tangent as an activation function.

Let's also change the value of the bias to be `6.8813735870195432` (so that we get derivative values with a small amount of digits after the comma) and display the computational graph.

**Test case:**

```console
python task13.py
```

![w02_06_result](../assets/w02_06_result.svg?raw=true "w02_06_result.svg")

## Task 14

**Description:**

Manually backpropagate the gradients.

**Test case:**

```console
python task14.py
```

![w02_07_result](../assets/w02_07_result.svg?raw=true "w02_07_result.svg")

## Task 15

**Description:**

Codify the differentiation process so that it can be executed automatically using a `backward` method that is called on the final (right-most node).

To do this, we'll need to:

- add another property to the `Value` class called `_backward` for automatic differentiation of the addition operation;
- define a function - `top_sort`, that accepts a list of `Value` objects and sort them topologically.
- integrate `top_sort` in a method called `backward` (notice that the `_backward` properties will stay).

**Acceptance criteria:**

1. The gradient for addition is calculated automatically.

**Test case:**

```console
python task15.py
```

![w02_08_result](../assets/w02_08_result.svg?raw=true "w02_08_result.svg")

## Task 16

**Description:**

Implement `_backward` for the multiplication and hyperbolic tangent operations.

**Acceptance criteria:**

1. The gradient for multiplication is calculated automatically.
2. The gradient for hyperbolic tangent application is calculated automatically.
3. The `backward` method is called from `main()`.

**Test case:**

```console
python task16.py
```

![w02_09_result](../assets/w02_09_result.svg?raw=true "w02_09_result.svg")

## Task 17

**Description:**

Currently, when we use a variable more than once the gradient gets overwritten. It can be seen below that the gradient of `x` should be `2` because `y = 2 * x`, but it is instead `1`.

![w02_10_result_bug](../assets/w02_10_result_bug.svg?raw=true "w02_10_result_bug.svg")

To fix this, we can accumulate the gradient instead of resetting it every time `_backward` is called.

**Acceptance criteria:**

1. The bug is fixed.

**Test case:**

```console
python task17.py
```

![w02_10_result](../assets/w02_10_result.svg?raw=true "w02_10_result.svg")

## Task 18

**Description:**

Extend the value class to allow the following operations:

- adding a float to a `Value` object;
- multiplying a `Value` object with a float;
- dividing a `Value` object by a float;
- exponentiation of a `Value` object with a float;
- exponentiation of Euler's number with a `Value` object.

We'll add the backpropagation (i.e. the implementation of the `_backward` function) in another task, so you needn't add it here.

**Test cases:**

```python
def main() -> None:
    x = Value(2.0, label='x')

    expected = Value(4.0)

    actuals = {
        'actual_sum_l': x + 2.0,
        'actual_sum_r': 2.0 + x,
        'actual_mul_l': x * 2.0,
        'actual_mul_r': 2.0 * x,
        'actual_div_r': (x + 6.0) / 2.0,
        'actual_pow_l': x**2,
        'actual_exp_e': x**2,
    }

    assert x.exp().data == np.exp(
        2), f"Mismatch for exponentiating Euler's number: expected {np.exp(2)}, but got {x.exp().data}."

    for actual_name, actual_value in actuals.items():
        assert actual_value.data == expected.data, f'Mismatch for {actual_name}: expected {expected.data}, but got {actual_value.data}.'

    print('All tests passed!')
```

```console
All tests passed!
```

## Task 19

**Description:**

Break down the hyperbolic tangent into the expressions that comprise it and backpropagate through them.

**Test case:**

```console
python task19.py
```

![w02_11_result](../assets/w02_11_result.svg?raw=true "w02_11_result.svg")
