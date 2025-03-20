# Table of Contents

- [Table of Contents](#table-of-contents)
- [Week 01 - Hello, Deep Learning. Implementing a Multilayer Perceptron](#week-01---hello-deep-learning-implementing-a-multilayer-perceptron)
  - [So what can deep learning models model?](#so-what-can-deep-learning-models-model)
  - [Modeling a neuron that can multiply by `2`](#modeling-a-neuron-that-can-multiply-by-2)
  - [Python Packages](#python-packages)
    - [Introduction](#introduction)
    - [Importing means executing `main.py`](#importing-means-executing-mainpy)
  - [NumPy](#numpy)
    - [Context](#context)
    - [Problems](#problems)
    - [Solution](#solution)
    - [Benefits](#benefits)
    - [Usage](#usage)
    - [2D NumPy Arrays](#2d-numpy-arrays)
    - [Basic Statistics](#basic-statistics)
    - [Generate data](#generate-data)
  - [Matplotlib](#matplotlib)
    - [Line plot](#line-plot)
    - [Scatter plot](#scatter-plot)
    - [Drawing multiple plots on one figure](#drawing-multiple-plots-on-one-figure)
    - [The logarithmic scale](#the-logarithmic-scale)
    - [Histogram](#histogram)
      - [Introduction](#introduction-1)
      - [In `matplotlib`](#in-matplotlib)
      - [Use cases](#use-cases)
    - [Checkpoint](#checkpoint)
    - [Customization](#customization)
      - [Axis labels](#axis-labels)
      - [Title](#title)
      - [Ticks](#ticks)
      - [Adding more data](#adding-more-data)
      - [`plt.tight_layout()`](#plttight_layout)
        - [Problem](#problem)
        - [Solution](#solution-1)
  - [Random numbers](#random-numbers)
    - [Context](#context-1)
    - [Random generators](#random-generators)
  - [A note on code formatting](#a-note-on-code-formatting)
- [Week 02 - Implementing Gradient Descent](#week-02---implementing-gradient-descent)
  - [Backpropagation](#backpropagation)
  - [Topological sort](#topological-sort)
  - [The hyperbolic tangent](#the-hyperbolic-tangent)
  - [Python OOP (Magic Methods)](#python-oop-magic-methods)
    - [Initialization and Construction](#initialization-and-construction)
    - [Arithmetic operators](#arithmetic-operators)
    - [String Magic Methods](#string-magic-methods)
    - [Comparison magic methods](#comparison-magic-methods)
- [Week 03 - Hello, PyTorch](#week-03---hello-pytorch)
  - [PyTorch. A deep learning framework](#pytorch-a-deep-learning-framework)
  - [Tensors. The building blocks of networks](#tensors-the-building-blocks-of-networks)
    - [What is a tensor?](#what-is-a-tensor)
    - [Creating tensors](#creating-tensors)
    - [Useful attributes](#useful-attributes)
    - [Shapes matter](#shapes-matter)
    - [Multiplication](#multiplication)
  - [Our first neural network using PyTorch](#our-first-neural-network-using-pytorch)
  - [Stacking layers with `nn.Sequential()`](#stacking-layers-with-nnsequential)
  - [Checkpoint](#checkpoint-1)
  - [Stacked linear transformations is still just one big linear transformation](#stacked-linear-transformations-is-still-just-one-big-linear-transformation)
  - [Sigmoid in PyTorch](#sigmoid-in-pytorch)
    - [Individually](#individually)
    - [As part of a network](#as-part-of-a-network)
  - [Softmax](#softmax)
  - [Checkpoint](#checkpoint-2)
  - [Training a network](#training-a-network)
  - [Cross-entropy loss in PyTorch](#cross-entropy-loss-in-pytorch)
  - [Minimizing the loss](#minimizing-the-loss)
    - [Backpropagation](#backpropagation-1)
    - [Optimizers](#optimizers)
  - [Putting it all together. Training a neural network](#putting-it-all-together-training-a-neural-network)
  - [Creating dataset and dataloader](#creating-dataset-and-dataloader)
  - [Gradients of the sigmoid and softmax functions](#gradients-of-the-sigmoid-and-softmax-functions)
  - [Introducing the **Re**ctified **L**inear **U**nit (`ReLU`)](#introducing-the-rectified-linear-unit-relu)
  - [Introducing Leaky ReLU](#introducing-leaky-relu)
  - [Counting the number of parameters](#counting-the-number-of-parameters)
    - [Layer naming conventions](#layer-naming-conventions)
    - [PyTorch's `numel` method](#pytorchs-numel-method)
    - [Checkpoint](#checkpoint-3)
  - [Learning rate and momentum](#learning-rate-and-momentum)
    - [Optimal learning rate](#optimal-learning-rate)
    - [Optimal momentum](#optimal-momentum)
  - [Layer initialization](#layer-initialization)
  - [Transfer learning](#transfer-learning)
    - [The goal](#the-goal)
    - [Fine-tuning](#fine-tuning)
    - [Checkpoint](#checkpoint-4)
  - [The Water Potability Dataset](#the-water-potability-dataset)
  - [Evaluating a model on a classification task](#evaluating-a-model-on-a-classification-task)
  - [Calculating validation loss](#calculating-validation-loss)
  - [The Bias-Variance Tradeoff](#the-bias-variance-tradeoff)
  - [Calculating accracy with `torchmetrics`](#calculating-accracy-with-torchmetrics)
  - [Fighting overfitting](#fighting-overfitting)
  - [Using a `Dropout` layer](#using-a-dropout-layer)
  - [Weight decay](#weight-decay)
  - [Data augmentation](#data-augmentation)
  - [Steps to maximize model performance](#steps-to-maximize-model-performance)
    - [Step 1: overfit the training set](#step-1-overfit-the-training-set)
    - [Step 2: reduce overfitting](#step-2-reduce-overfitting)
    - [Step 3: fine-tune hyperparameters](#step-3-fine-tune-hyperparameters)
- [Week 04 - Convolutional Neural Networks. Building multi-input and multi-output models](#week-04---convolutional-neural-networks-building-multi-input-and-multi-output-models)
  - [Custom PyTorch Datasets](#custom-pytorch-datasets)
  - [Checkpoint](#checkpoint-5)
  - [Class-Based PyTorch Model](#class-based-pytorch-model)
  - [Unstable gradients](#unstable-gradients)
    - [Solutions to unstable gradients](#solutions-to-unstable-gradients)
      - [Proper weights initialization](#proper-weights-initialization)
      - [Batch normalization](#batch-normalization)
  - [The Clouds dataset](#the-clouds-dataset)
  - [Converting pixels to tensors and tensors to pixels](#converting-pixels-to-tensors-and-tensors-to-pixels)
    - [`ToTensor()`](#totensor)
    - [`PILToTensor()`](#piltotensor)
    - [`ToPILImage()`](#topilimage)
  - [Loading images with PyTorch](#loading-images-with-pytorch)
  - [Data augmentation](#data-augmentation-1)
  - [CNNs - The neural networks for image processing](#cnns---the-neural-networks-for-image-processing)
  - [Architecture](#architecture)
  - [Precision \& Recall for Multiclass Classification (revisited)](#precision--recall-for-multiclass-classification-revisited)
    - [Computing total value](#computing-total-value)
    - [Computing per class value](#computing-per-class-value)
  - [Multi-input models](#multi-input-models)
  - [The Omniglot dataset](#the-omniglot-dataset)
  - [Multi-output models](#multi-output-models)
  - [Character and alphabet classification](#character-and-alphabet-classification)
  - [Loss weighting](#loss-weighting)
    - [Varying task importance](#varying-task-importance)
    - [Losses on different scales](#losses-on-different-scales)
  - [Checkpoint](#checkpoint-6)
- [Week 05 - Image Processing](#week-05---image-processing)
  - [Introduction](#introduction-2)
    - [Context](#context-2)
    - [Problem](#problem-1)
    - [Solution](#solution-2)
    - [Benefits](#benefits-1)
    - [Context](#context-3)
    - [Problem](#problem-2)
    - [Solution - scikit-image](#solution---scikit-image)
    - [Images in scikit-image](#images-in-scikit-image)
  - [RGB and Grayscale](#rgb-and-grayscale)
    - [Context](#context-4)
    - [Problem](#problem-3)
    - [Solution - `color.rgb2gray` and `color.gray2rgb`](#solution---colorrgb2gray-and-colorgray2rgb)
  - [Basic image operations](#basic-image-operations)
    - [Using `numpy`](#using-numpy)
      - [Vertical flip](#vertical-flip)
      - [Horizontal flip](#horizontal-flip)
    - [The `transform` module](#the-transform-module)
      - [Rotating](#rotating)
      - [Rescaling](#rescaling)
      - [Resizing](#resizing)
        - [Problem](#problem-4)
        - [Solution - anti-aliasing](#solution---anti-aliasing)
  - [Thresholding](#thresholding)
    - [Context](#context-5)
    - [Problem](#problem-5)
    - [Simple solution - global / histogram based thresholding](#simple-solution---global--histogram-based-thresholding)
    - [Advanced solutions: local / adaptive thresholding](#advanced-solutions-local--adaptive-thresholding)
  - [Edge detection](#edge-detection)
    - [Context](#context-6)
    - [Problem](#problem-6)
    - [Solution - Sobel filter](#solution---sobel-filter)
    - [Advanced solution - edge detection with the Canny algorithm](#advanced-solution---edge-detection-with-the-canny-algorithm)
  - [Contrast enhancement](#contrast-enhancement)
    - [Context](#context-7)
    - [Problem](#problem-7)
    - [Solutions](#solutions)
      - [Standard Histogram Equalization](#standard-histogram-equalization)
      - [Contrastive Limited Adaptive Equalization](#contrastive-limited-adaptive-equalization)
    - [Checkpoint](#checkpoint-7)
  - [Image Morphology](#image-morphology)
  - [Checkpoint](#checkpoint-8)
  - [Shapes in `scikit-image`](#shapes-in-scikit-image)
  - [Restoring images with inpainting](#restoring-images-with-inpainting)
  - [Denoising images](#denoising-images)
    - [Simple solution - Gaussian smoothing](#simple-solution---gaussian-smoothing)
    - [Advanced solutions](#advanced-solutions)
  - [Segmentation](#segmentation)
    - [Superpixels](#superpixels)
    - [Simple Linear Iterative Clustering (SLIC)](#simple-linear-iterative-clustering-slic)
  - [Image contours](#image-contours)
  - [Corner detection](#corner-detection)
    - [Corners](#corners)
    - [Harris corner detector](#harris-corner-detector)
  - [Face detection](#face-detection)
  - [Applications](#applications)

# Week 01 - Hello, Deep Learning. Implementing a Multilayer Perceptron

What is your experience with deep learning? Who has built a deep learning model? What was it about?

<details>

<summary>What is deep learning?</summary>

- Deep learning is a class of algorithms that solve the task of `automatic pattern recognition`.
- There are two main paradigms of programming: `imperative` and `functional`. Deep learning can be regarded as a third paradigm, different from the other two as follows: let's say you have a particular `task` you want to solve.
  - In imperative and functional programming, `you write the code directly`; you tell the machine what it has to do **explicitly** and you write exactly the code solves the task by outlining and connecting multiple steps together (i.e. **you** create an algorithm).
  - In deep learning you are **not** explicitly / directly writing the logic that would solve the task. Instead, you build a `deep learning model` that models the task you are tying to solve and **the model itself creates the algorithm** for solving your task.
- A deep learning model is a set of parameters connected in various ways. It solves tasks by finding optimal values of those parameters - i.e. values for which it can in **most** cases solve a task. The word **most** is important - notice that all a deep learning model is an `automatic mathematical optimization model` for a set of parameters. **`It solves tasks by approximation, not by building explicit logic`**.
- The process during which the model optimizes its parameters is called `training`.

</details>

<details>

<summary>How are deep learning models built?</summary>

Deep learning models are built by codifying the `description` of the desired model behavior. This description of the expected behavior is `implicitly` hidden in the data in the form of `patterns`. Thus, deep learning is uncovering those patterns and using them to solve various problems.

The process is called `training` - you give the untrained model your data (your `description` of desired behavior) and the model "tweaks" its parameters until it fits your description well enough. And there you have it - a deep learning model that does what you want (probably 😄 (i.e. with a certain probability, because it's never going to be perfect)).

You can think about the "tweaking" process as the process in which multiple models are created each with different values for their parameters, their accuracies are compared and the model with the highest accuracy is chosen as the final version.

</details>

## So what can deep learning models model?

Here is the twist - `everything`! For any task as long as you have enough data, you can model it.

One of the things you can model is **the probability of the next word in a sentence**. Surprise - the models that solve these types of tasks are called `Large Language Models`! You have a partially written sentence and you can create a mathematical model that predicts how likely every possible word, in the language you're working in, is to be the next word in that sentence. And after you've selected the word, you can repeat the process on this extended sentence - that's how you get `ChatGPT`.

## Modeling a neuron that can multiply by `2`

![w01_multiplier.png](assets/w01_multiplier.png "w01_multiplier.png")

We want to teach the model that `w` has to equal `2`.

<details>

<summary>How would we go about doing this?</summary>

1. We start with a random guess for `w`. For the sake of concreteness, let's say a random floating-point number in the interval `[0, 10)`. The interval does not matter  - even if it does not contain `2` the traning process would converge towards it.

![w01_multiplier_guess.png](assets/w01_multiplier_guess.png "w01_multiplier_guess.png")

2. We calculate the value of the loss function at that initial random guess.

![w01_multiplier_loss.png](assets/w01_multiplier_loss.png "w01_multiplier_loss.png")

3. We can see what will happen if we "wiggle" `w` by a tiny amount `eps`.

![w01_multiplier_loss_wiggle.png](assets/w01_multiplier_loss_wiggle.png "w01_multiplier_loss_wiggle.png")

4. So, the value either goes up or down. This means that our loss function would represent a parabola.

![w01_multiplier_loss_viz.png](assets/w01_multiplier_loss_viz.png "w01_multiplier_loss_viz.png")

5. If only we had a way to always know in which direction the value would go down? Oh, wait - we do! It's the opposite direction of the one in which the derivative grows!

![w01_derivative_values.gif](assets/w01_derivative_values.gif "w01_derivative_values.gif")

For now, we won't calculate the exact derivative because we don't need to do that - we can use its general formula:

$${\displaystyle L=\lim _{eps\to 0}{\frac {loss(w+eps)-loss(w)}{eps}}}$$

6. We can then use `L` to step in the direction of decline, by doing: `w -= L`.

7. This, however, will have a problem: the value of `L` might be very high. If our step is always `L` we would start oscilating. Therefore, we'll use a learning rate that will say how large our step would be: `w -= learning_rate * L`.

And this is it! This process is guaranteed to find `2` as the optimal value. Moreover, this iterative algorithm for minimizing a differentiable multivariate function is what is also known as [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent) 😇.

</details>

<details>

<summary>What would the architecture and process for creating a model of an "AND" logical gate look like?</summary>

We might start off with something like this:

![w01_and_or_models.png](assets/w01_and_or_models.png "w01_and_or_models.png")

However, because our task now shifts from being a regression one into a classification one, we can also add the `sigmoid` function to control the output values:

$${\displaystyle f(x)={\frac {1}{1+e^{-x}}}}$$

![w01_and_or_models_sigmoid.png](assets/w01_and_or_models_sigmoid.png "w01_and_or_models_sigmoid.png")

<details>

<summary>But! Adding the sigmoid activation function actually causes another problem - for what values of w1 and w2 would we have a problem?</summary>

Look at what happens when we have $w_1=0$ and $w_2=0$ (our model is guessing correctly that the output should be `0`):

![w01_sigmoid_problem.png](assets/w01_sigmoid_problem.png "w01_sigmoid_problem.png")

</details>

<details>

<summary>How do we fix this?</summary>

We need to keep the weights at `0` but also add another term that can control the logit value when all weights are `0`. Welcome, ***bias***.

![w01_bias.png](assets/w01_bias.png "w01_bias.png")

</details>

</details>

<details>

<summary>How do we model the "XOR" logical gate?</summary>

Let's see how the classes are distributed in `2D` space:

![w01_class_distribution.png](assets/w01_class_distribution.png "w01_class_distribution.png")

The models we defined above are actually called perceptrons. They calculate a weighted sum of their inputs and thresholds it with a step function.

Geometrically, this means **the perceptron can separate its input space with a hyperplane**. That’s where the notion that a perceptron can only separate linearly separable problems comes from.

Since the `XOR` function **is not linearly separable**, it really is impossible for a single hyperplane to separate it.

<details>

<summary>What are our next steps then?</summary>

We need to describe the `XOR` gate using non-`XOR` gates. This can be done:

`(x|y) & ~(x&y)`

So, the `XOR` model can then be represented using the following architecture:

![w01_xor_architecture.png](assets/w01_xor_architecture.png "w01_xor_architecture.png")

<details>

<summary>How many parameters would we have in total?</summary>

9

</details>

</details>

</details>

## Python Packages

### Introduction

You write all of your code to one and the same Python script.

<details>

<summary>What are the problems that arise from that?</summary>

- Huge code base: messy;
- Lots of code you won't use;
- Maintenance problems.

</details>

<details>

<summary>How do we solve this problem?</summary>

We can split our code into libraries (or in the Python world - **packages**).

Packages are a directory of Python scripts.

Each such script is a so-called **module**.

Here's the hierarchy visualized:

![w01_packages_modules.png](./assets/w01_packages_modules.png "w01_packages_modules.png")

These modules specify functions, methods and new Python types aimed at solving particular problems. There are thousands of Python packages available from the Internet. Among them are packages for data science:

- there's **NumPy to efficiently work with arrays**;
- **Matplotlib for data visualization**;
- **scikit-learn for machine learning**.

</details>

Not all of them are available in Python by default, though. To use Python packages, you'll first have to install them on your own system, and then put code in your script to tell Python that you want to use these packages. Advice:

- always install packages in **virtual environments** (abstractions that hold packages for separate projects).
  - You can create a virtual environment by using the following code:

    ```console
    python3 -m venv .venv
    ```

    This will create a hidden folder, called `.venv`, that will store all packages you install for your current project (instead of installing them globally on your system).

  - If there is a `requirements.txt` file, use it to install the needed packages beforehand.
    - In the github repo, there is such a file - you can use it to install all the packages you'll need in the course. This can be done by using this command:

    ```console
    (if on Windows) > .venv\Scripts\activate
    (if on Linux) > source .venv/bin/activate
    (.venv) > pip install -r requirements.txt
    ```

Now that the package is installed, you can actually start using it in one of your Python scripts. To do this you should import the package, or a specific module of the package.

You can do this with the `import` statement. To import the entire `numpy` package, you can do `import numpy`. A commonly used function in NumPy is `array`. It takes a Python list as input and returns a [`NumPy array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html) object as an output. The NumPy array is very useful to do data science, but more on that later. Calling the `array` function like this, though, will generate an error:

```python
import numpy
array([1, 2, 3])
```

```console
NameError: name `array` is not defined
```

To refer to the `array` function from the `numpy` package, you'll need this:

```python
import numpy
numpy.array([1, 2, 3])
```

```console
array([1, 2, 3])
```

This time it works.

Using this `numpy.` prefix all the time can become pretty tiring, so you can also import the package and refer to it with a different name. You can do this by extending your `import` statement with `as`:

```python
import numpy as np
np.array([1, 2, 3])
```

```console
array([1, 2, 3])
```

Now, instead of `numpy.array`, you'll have to use `np.array` to use NumPy's functions.

There are cases in which you only need one specific function of a package. Python allows you to make this explicit in your code.

Suppose that we ***only*** want to use the `array` function from the NumPy package. Instead of doing `import numpy`, you can instead do `from numpy import array`:

```python
from numpy import array
array([1, 2, 3])
```

```console
array([1, 2, 3])
```

This time, you can simply call the `array` function without `numpy.`.

This `from import` version to use specific parts of a package can be useful to limit the amount of coding, but you're also loosing some of the context. Suppose you're working in a long Python script. You import the array function from numpy at the very top, and way later, you actually use this array function. Somebody else who's reading your code might have forgotten that this array function is a specific NumPy function; it's not clear from the function call.

![w01_from_numpy.png](./assets/w01_from_numpy.png "w01_from_numpy.png")

^ using numpy, but not very clear

Thus, the more standard `import numpy as np` call is preferred: In this case, your function call is `np.array`, making it very clear that you're working with NumPy.

![w01_import_as_np.png](./assets/w01_import_as_np.png "w01_import_as_np.png")

- Suppose you want to use the function `inv()`, which is in the `linalg` subpackage of the `scipy` package. You want to be able to use this function as follows:

    ```python
    my_inv([[1,2], [3,4]])
    ```

    Which import statement will you need in order to run the above code without an error?

  - A. `import scipy`
  - B. `import scipy.linalg`
  - C. `from scipy.linalg import my_inv`
  - D. `from scipy.linalg import inv as my_inv`

    <details>

    <summary>Reveal answer:</summary>

    Answer: D

    </details>

### Importing means executing `main.py`

Remember that importing a package is equivalent to executing everything in the `main.py` module. Thus. you should always have `if __name__ == '__main__'` block of code and call your functions from there.

Run the scripts `test_script1.py` and `test_script2.py` to see the differences.

## NumPy

### Context

Python lists are pretty powerful:

- they can hold a collection of values with different types (heterogeneous data structure);
- easy to change, add, remove elements;
- many built-in functions and methods.

### Problems

This is wonderful, but one feature is missing, a feature that is super important for aspiring data scientists and machine learning engineers - carrying out mathematical operations **over entire collections of values** and doing it **fast**.

Let's take the heights and weights of your family and yourself. You end up with two lists, `height`, and `weight` - the first person is `1.73` meters tall and weighs `65.4` kilograms and so on.

```python
height = [1.73, 1.68, 1.71, 1.89, 1.79]
height
```

```console
[1.73, 1.68, 1.71, 1.89, 1.79]
```

```python
weight = [65.4, 59.2, 63.6, 88.4, 68.7]
weight
```

```console
[65.4, 59.2, 63.6, 88.4, 68.7]
```

If you now want to calculate the Body Mass Index for each family member, you'd hope that this call can work, making the calculations element-wise. Unfortunately, Python throws an error, because it has no idea how to do calculations on lists. You could solve this by going through each list element one after the other, and calculating the BMI for each person separately, but this is terribly inefficient and tiresome to write.

### Solution

- `NumPy`, or Numeric Python;
- Provides an alternative to the regular Python list: the NumPy array;
- The NumPy array is pretty similar to the list, but has one additional feature: you can perform calculations over entire arrays;
- super-fast as it's based on C++
- Installation:
  - In the terminal: `pip install numpy`

### Benefits

Speed, speed, speed:

- Stackoverflow: <https://stackoverflow.com/questions/73060352/is-numpy-any-faster-than-default-python-when-iterating-over-a-list>
- Visual Comparison:

    ![w01_np_vs_list.png](./assets/w01_np_vs_list.png "w01_np_vs_list.png")

### Usage

```python
import numpy as np
np_height = np.array(height)
np_height
```

```console
array([1.73, 1.68, 1.71, 1.89, 1.79])
```

```python
import numpy as np
np_weight = np.array(weight)
np_weight
```

```console
array([65.4, 59.2, 63.6, 88.4, 68.7])
```

```python
# Calculations are performed element-wise.
# 
# The first person's BMI was calculated by dividing the first element in np_weight
# by the square of the first element in np_height,
# 
# the second person's BMI was calculated with the second height and weight elements, and so on.
bmi = np_weight / np_height ** 2
bmi
```

```console
array([21.851, 20.975, 21.750, 24.747, 21.441])
```

in comparison, the above will not work for Python lists:

```python
weight / height ** 2
```

```console
TypeError: unsupported operand type(s) for ** or pow(): 'list' and 'int'
```

You should still pay attention, though:

- `numpy` assumes that your array contains values **of a single type**;
- a NumPy array is simply a new kind of Python type, like the `float`, `str` and `list` types. This means that it comes with its own methods, which can behave differently than you'd expect.

    ```python
    python_list = [1, 2, 3]
    numpy_array = np.array([1, 2, 3])
    ```

    ```python
    python_list + python_list
    ```

    ```console
    [1, 2, 3, 1, 2, 3]
    ```

    ```python
    numpy_array + numpy_array
    ```

    ```console
    array([2, 4, 6])
    ```

- When you want to get elements from your array, for example, you can use square brackets as with Python lists. Suppose you want to get the bmi for the second person, so at index `1`. This will do the trick:

    ```python
    bmi
    ```

    ```console
    [21.851, 20.975, 21.750, 24.747, 21.441]
    ```

    ```python
    bmi[1]
    ```

    ```console
    20.975
    ```

- Specifically for NumPy, there's also another way to do list subsetting: using an array of booleans.

    Say you want to get all BMI values in the bmi array that are over `23`.

    A first step is using the greater than sign, like this: The result is a NumPy array containing booleans: `True` if the corresponding bmi is above `23`, `False` if it's below.

    ```python
    bmi > 23
    ```

    ```console
    array([False, False, False, True, False])
    ```

    Next, you can use this boolean array inside square brackets to do subsetting. Only the elements in `bmi` that are above `23`, so for which the corresponding boolean value is `True`, is selected. There's only one BMI that's above `23`, so we end up with a NumPy array with a single value, that specific BMI. Using the result of a comparison to make a selection of your data is a very common way to work with data.

    ```python
    bmi[bmi > 23]
    ```

    ```console
    array([24.747])
    ```

### 2D NumPy Arrays

If you ask for the type of these arrays, Python tells you that they are `numpy.ndarray`:

```python
np_height = np.array([1.73, 1.68, 1.71, 1.89, 1.79])
np_weight = np.array([65.4, 59.2, 63.6, 88.4, 68.7])
type(np_height)
```

```console
numpy.ndarray
```

```python
type(np_weight)
```

```console
numpy.ndarray
```

`ndarray` stands for n-dimensional array. The arrays `np_height` and `np_weight` are one-dimensional arrays, but it's perfectly possible to create `2`-dimensional, `3`-dimensional and `n`-dimensional arrays.

You can create a 2D numpy array from a regular Python list of lists:

```python
np_2d = np.array([[1.73, 1.68, 1.71, 1.89, 1.79],
                  [65.4, 59.2, 63.6, 88.4, 68.7]])
np_2d
```

```console
array([[ 1.73,  1.68,  1.71,  1.89,  1.79],
       [65.4 , 59.2 , 63.6 , 88.4 , 68.7 ]])
```

Each sublist in the list, corresponds to a row in the `2`-dimensional numpy array. Using `.shape`, you can see that we indeed have `2` rows and `5` columns:

```python
np_2d.shape
```

```console
(2, 5) # 2 rows, 5 columns
```

`shape` is a so-called **attribute** of the `np2d` array, that can give you more information about what the data structure looks like.

> **Note:** The syntax for accessing an attribute looks a bit like calling a method, but they are not the same! Remember that methods have round brackets (`()`) after them, but attributes do not.
>
> **Note:** For n-D arrays, the NumPy rule still applies: an array can only contain a single type.

You can think of the 2D numpy array as a faster-to-work-with list of lists: you can perform calculations and more advanced ways of subsetting.

Suppose you want the first row, and then the third element in that row - you can grab it like this:

```python
np_2d[0][2]
```

```console
1.71
```

or use an alternative way of subsetting, using single square brackets and a comma:

```python
np_2d[0, 2]
```

```console
1.71
```

The value before the comma specifies the row, the value after the comma specifies the column. The intersection of the rows and columns you specified, are returned. This is the syntax that's most popular.

Suppose you want to select the height and weight of the second and third family member from the following array.

```console
array([[ 1.73,  1.68,  1.71,  1.89,  1.79],
       [65.4 , 59.2 , 63.6 , 88.4 , 68.7 ]])
```

<details>

<summary>How can this be achieved?</summary>

Answer: np_2d[:, [1, 2]]

</details>

### Basic Statistics

A typical first step in analyzing your data, is getting to know your data in the first place.

Imagine you conduct a city-wide survey where you ask `5000` adults about their height and weight. You end up with something like this: a 2D numpy array, that has `5000` rows, corresponding to the `5000` people, and `2` columns, corresponding to the height and the weight.

```python
np_city = ...
np_city
```

```console
array([[ 2.01, 64.33],
       [ 1.78, 67.56],
       [ 1.71, 49.04],
       ...,
       [ 1.73, 55.37],
       [ 1.72, 69.73],
       [ 1.85, 66.69]])
```

Simply staring at these numbers, though, won't give you any insights. What you can do is generate summarizing statistics about them.

- you can try to find out the average height of these people, with NumPy's `mean` function:

```python
np.mean(np_city[:, 0]) # alternative: np_city[:, 0].mean()
```

```console
1.7472
```

It appears that on average, people are `1.75` meters tall.

- What about the median height? This is the height of the middle person if you sort all persons from small to tall. Instead of writing complicated python code to figure this out, you can simply use NumPy's `median` function:

```python
np.median(np_city[:, 0]) # alternative: np_city[:, 0].median()
```

```console
1.75
```

You can do similar things for the `weight` column in `np_city`. Often, these summarizing statistics will provide you with a "sanity check" of your data. If you end up with a average weight of `2000` kilograms, your measurements are most likely incorrect. Apart from mean and median, there's also other functions, like:

```python
np.corrcoef(np_city[:, 0], np_city[:, 1])
```

```console
array([[1.       , 0.0082912],
       [0.0082912, 1.       ]])
```

```python
np.std(np_city[:, 0])
```

```console
np.float64(0.19759467357193614)
```

`sum()`, `sort()`, etc, etc. See all of them [here](https://numpy.org/doc/stable/reference/routines.statistics.html).

### Generate data

The data used above was generated using the following code. Two random distributions were sampled 5000 times to create the `height` and `weight` arrays, and then `column_stack` was used to paste them together as two columns.

```python
import numpy as np
height = np.round(np.random.normal(1.75, 0.2, 5000), 2)
weight = np.round(np.random.normal(60.32, 15, 5000), 2)
np_city = np.column_stack((height, weight))
```

## Matplotlib

The better you understand your data, the better you'll be able to extract insights. And once you've found those insights, again, you'll need visualization to be able to share your valuable insights with other people.

![w01_matplotlib.png](./assets/w01_matplotlib.png "w01_matplotlib.png")

There are many visualization packages in python, but the mother of them all, is `matplotlib`. You will need its subpackage `pyplot`. By convention, this subpackage is imported as `plt`:

```python
import matplotlib.pyplot as plt
```

### Line plot

Let's try to gain some insights in the evolution of the world population. To plot data as a **line chart**, we call `plt.plot` and use our two lists as arguments. The first argument corresponds to the horizontal axis, and the second one to the vertical axis.

```python
year = [1950, 1970, 1990, 2010]
pop = [2.519, 3.692, 5.263, 6.972]

# "plt.plot" creates the plot, but does not display it
plt.plot(year, pop)

# "plt.show" displays the plot
plt.show()
```

You'll have to call `plt.show()` explicitly because you might want to add some extra information to your plot before actually displaying it, such as titles and label customizations.

As a result we get:

![w01_matplotlib_result.png](./assets/w01_matplotlib_result.png "w01_matplotlib_result.png")

We see that:

- the years are indeed shown on the horizontal axis;
- the populations on the vertical axis;
- this type of plot is great for plotting a time scale along the x-axis and a numerical feature on the y-axis.

There are four data points, and Python draws a line between them.

![w01_matplotlib_edited.png](./assets/w01_matplotlib_edited.png "w01_matplotlib_edited.png")

In 1950, the world population was around 2.5 billion. In 2010, it was around 7 billion.

> **Insight:** The world population has almost tripled in sixty years.
>
> **Note:** If you pass only one argument to `plt.plot`, Python will know what to do and will use the index of the list to map onto the `x` axis, and the values in the list onto the `y` axis.

### Scatter plot

We can reuse the code from before and just swap `plt.plot(...)` with `plt.scatter(...)`:

```python
year = [1950, 1970, 1990, 2010]
pop = [2.519, 3.692, 5.263, 6.972]

# "plt.plot" creates the plot, but does not display it
plt.scatter(year, pop)

# "plt.show" displays the plot
plt.show()
```

![w01_matplotlib_scatter.png](./assets/w01_matplotlib_scatter.png "w01_matplotlib_scatter.png")

The resulting scatter plot:

- plots the individual data points;
- dots aren't connected with a line;
- is great for plotting two numerical features (example: correlation analysis).

### Drawing multiple plots on one figure

This can be done by first instantiating the figure and two axis and the using each axis to plot the data. Example taken from [here](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots).

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
f.suptitle('Sharing Y axis')

ax1.plot(x, y)
ax2.scatter(x, y)

plt.show()
```

![w01_multiplot.png](./assets/w01_multiplot.png "w01_multiplot.png")

### The logarithmic scale

Sometimes the correlation analysis between two variables can be done easier when one or all of them is plotted on a logarithmic scale. This is because we would reduce the difference between large values as this scale "squashes" large numbers:

![w01_logscale.png](./assets/w01_logscale.png "w01_logscale.png")

In `matplotlib` we can use the [plt.xscale](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xscale.html) function to change the scaling of an axis using `plt` or [ax.set_xscale](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xscale.html#matplotlib.axes.Axes.set_xscale) to set the scale of an axis of a subplot.

### Histogram

#### Introduction

The histogram is a plot that's useful to explore **distribution of numeric** data;

Imagine `12` values between `0` and `6`.

![w01_histogram_ex1.png](./assets/w01_histogram_ex1.png "w01_histogram_ex1.png")

To build a histogram for these values, you can divide the line into **equal chunks**, called **bins**. Suppose you go for `3` bins, that each have a width of `2`:

![w01_histogram_ex2.png](./assets/w01_histogram_ex2.png "w01_histogram_ex2.png")

Next, you count how many data points sit inside each bin. There's `4` data points in the first bin, `6` in the second bin and `2` in the third bin:

![w01_histogram_ex3.png](./assets/w01_histogram_ex3.png "w01_histogram_ex3.png")

Finally, you draw a bar for each bin. The height of the bar corresponds to the number of data points that fall in this bin. The result is a histogram, which gives us a nice overview on how the `12` values are **distributed**. Most values are in the middle, but there are more values below `2` than there are above `4`:

![w01_histogram_ex4.png](./assets/w01_histogram_ex4.png "w01_histogram_ex4.png")

#### In `matplotlib`

In `matplotlib` we can use the `.hist` function. In its documentation there're a bunch of arguments you can specify, but the first two are the most used ones:

- `x` should be a list of values you want to build a histogram for;
- `bins` is the number of bins the data should be divided into. Based on this number, `.hist` will automatically find appropriate boundaries for all bins, and calculate how may values are in each one. If you don't specify the bins argument, it will by `10` by default.

![w01_histogram_matplotlib.png](./assets/w01_histogram_matplotlib.png "w01_histogram_matplotlib.png")

The number of bins is important in the following way:

- too few bins will oversimplify reality and won't show you the details;
- too many bins will overcomplicate reality and won't show the bigger picture.

Experimenting with different numbers and/or creating multiple plots on the same canvas can alleviate that.

Here's the code that generated the above example:

```python
import matplotlib.pyplot as plt
xs = [0, 0.6, 1.4, 1.6, 2.2, 2.5, 2.6, 3.2, 3.5, 3.9, 4.2, 6]
plt.hist(xs, bins=3)
plt.show()
```

and the result of running it:

![w01_histogram_matplotlib_code.png](./assets/w01_histogram_matplotlib_code.png "w01_histogram_matplotlib_code.png")

#### Use cases

Histograms are really useful to give a bigger picture. As an example, have a look at this so-called **population pyramid**. The age distribution is shown, for both males and females, in the European Union.

![w01_population_pyramid.png](./assets/w01_population_pyramid.png "w01_population_pyramid.png")

Notice that the histograms are flipped 90 degrees; the bins are horizontal now. The bins are largest for the ages `40` to `44`, where there are `20` million males and `20` million females. They are the so called baby boomers. These are figures of the year `2010`. What do you think will have changed in `2050`?

Let's have a look.

![w01_population_pyramid_full.png](./assets/w01_population_pyramid_full.png "w01_population_pyramid_full.png")

The distribution is flatter, and the baby boom generation has gotten older. **With the blink of an eye, you can easily see how demographics will be changing over time.** That's the true power of histograms at work here!

### Checkpoint

<details>

<summary>
You want to visually assess if the grades on your exam follow a particular distribution. Which plot do you use?

```text
A. Line plot.
B. Scatter plot.
C. Histogram.
```

</summary>

Answer: C.

</details>

<details>

<summary>
You want to visually assess if longer answers on exam questions lead to higher grades. Which plot do you use?

```text
A. Line plot.
B. Scatter plot.
C. Histogram.
```

</summary>

Answer: B.

</details>

### Customization

Creating a plot is one thing. Making the correct plot, that makes the message very clear - that's the real challenge.

For each visualization, you have many options:

- change colors;
- change shapes;
- change labels;
- change axes, etc., etc.

The choice depends on:

- the data you're plotting;
- the story you want to tell with this data.

Below are outlined best practices when it comes to creating an MVP plot.

If we run the script for creating a line plot, we already get a pretty nice plot:

![w01_plot_basic.png](./assets/w01_plot_basic.png "w01_plot_basic.png")

It shows that the population explosion that's going on will have slowed down by the end of the century.

But some things can be improved:

- **axis labels**;
- **title**;
- **ticks**.

#### Axis labels

The first thing you always need to do is label your axes. We can do this by using the `xlabel` and `ylabel` functions. As inputs, we pass strings that should be placed alongside the axes.

![w01_plot_axis_labels.png](./assets/w01_plot_axis_labels.png "w01_plot_axis_labels.png")

#### Title

We're also going to add a title to our plot, with the `title` function. We pass the actual title, `'World Population Projections'`, as an argument:

![w01_plot_title.png](./assets/w01_plot_title.png "w01_plot_title.png")

#### Ticks

Using `xlabel`, `ylabel` and `title`, we can give the reader more information about the data on the plot: now they can at least tell what the plot is about.

To put the population growth in perspective, the y-axis should start from `0`. This can be achieved by using the `yticks` function. The first input is a list, in this example with the numbers `0` up to `10`, with intervals of `2`:

![w01_plot_ticks.png](./assets/w01_plot_ticks.png "w01_plot_ticks.png")

Notice how the curve shifts up. Now it's clear that already in `1950`, there were already about `2.5` billion people on this planet.

Next, to make it clear we're talking about billions, we can add a second argument to the `yticks` function, which is a list with the display names of the ticks. This list should have the same length as the first list.

![w01_plot_tick_labels.png](./assets/w01_plot_tick_labels.png "w01_plot_tick_labels.png")

#### Adding more data

Finally, let's add some more historical data to accentuate the population explosion in the last `60` years. If we run the script once more, three data points are added to the graph, giving a more complete picture.

![w01_plot_more_data.png](./assets/w01_plot_more_data.png "w01_plot_more_data.png")

#### `plt.tight_layout()`

##### Problem

With the default Axes positioning, the axes title, axis labels, or tick labels can sometimes go outside the figure area, and thus get clipped.

```python
import matplotlib.pyplot as plt
import numpy as np

def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

fig, ax = plt.subplots()
example_plot(ax, fontsize=24)
plt.show()
```

![w01_tight_layout_1.png](./assets/w01_tight_layout_1.png "w01_tight_layout_1.png")

##### Solution

To prevent this, the location of Axes needs to be adjusted. `plt.tight_layout()` does this automatically:

```python
import matplotlib.pyplot as plt
import numpy as np

def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

fig, ax = plt.subplots()
example_plot(ax, fontsize=24)
plt.tight_layout()
plt.show()
```

![w01_tight_layout_2.png](./assets/w01_tight_layout_2.png "w01_tight_layout_2.png")

When you have multiple subplots, often you see labels of different Axes overlapping each other:

```python
import matplotlib.pyplot as plt
import numpy as np

def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)
plt.show()
```

![w01_tight_layout_3.png](./assets/w01_tight_layout_3.png "w01_tight_layout_3.png")

`plt.tight_layout()` will also adjust spacing between subplots to minimize the overlaps:

```python
import matplotlib.pyplot as plt
import numpy as np

def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)
plt.tight_layout()
plt.show()
```

![w01_tight_layout_4.png](./assets/w01_tight_layout_4.png "w01_tight_layout_4.png")

## Random numbers

### Context

Imagine the following:

- you're walking up the empire state building and you're playing a game with a friend.
- You throw a die `100` times:
  - If it's `1` or `2` you'll go one step down.
  - If it's `3`, `4`, or `5`, you'll go one step up.
  - If you throw a `6`, you'll throw the die again and will walk up the resulting number of steps.
- also, you admit that you're a bit clumsy and have a chance of `0.1%` of falling down the stairs when you make a move. Falling down means that you have to start again from step `0`.

With all of this in mind, you bet with your friend that you'll reach `60` steps high. What is the chance that you will win this bet?

- one way to solve it would be to calculate the chance analytically using equations;
- another possible approach, is to simulate this process thousands of times, and see in what fraction of the simulations that you will reach `60` steps.

We're going to opt for the second approach.

### Random generators

We have to simulate the die. To do this, we can use random generators.

```python
import numpy as np
np.random.rand() # Pseudo-random numbers
```

```console
0.026360555982748446
```

We get a random number between `0` and `1`. This number is so-called pseudo-random. Those are random numbers that are generated using a mathematical formula, starting from a **random seed**.

This seed was chosen by Python when we called the `rand` function, but you can also set this manually. Suppose we set it to `123` and then call the `rand` function twice.

```python
import numpy as np
np.random.seed(123)
print(np.random.rand())
print(np.random.rand())
```

```console
0.6964691855978616
0.28613933495037946
```

> **Note:** Set the seed in the global scope of the Python module (not in a function).

We get two random numbers, however, if call `rand` twice more ***from a new python session***, we get the exact same random numbers!

```python
import numpy as np
np.random.seed(123)
print(np.random.rand())
print(np.random.rand())
```

```console
0.6964691855978616
0.28613933495037946
```

This is funky: you're generating random numbers, but for the same seed, you're generating the same random numbers. That's why it's called pseudo-random; **it's random but consistent between runs**; this is very useful, because this ensures ***"reproducibility"***. Other people can reproduce your analysis.

Suppose we want to simulate a coin toss.

- we set the seed;
- we use the `np.random.randint()` function: it will randomly generate either `0` or `1`. We'll pass two arguments to determine the range of the generated numbers - `0` and `2` (non-inclusive on the right side).

```python
import numpy as np
np.random.seed(123)
print(np.random.randint(0, 2))
print(np.random.randint(0, 2))
print(np.random.randint(0, 2))
```

```console
0
1
0
```

We can extend the code with an `if-else` statement to improve user experience:

```python
import numpy as np
np.random.seed(123)
coin = np.random.randint(0, 2)
print(coin)
if coin == 0:
    print('heads')
else:
    print('tails')
```

```console
heads
```

## A note on code formatting

In this course we'll strive to learn how to develop scripts in Python. In general, good code in software engineering is one that is:

1. Easy to read.
2. Safe from bugs.
3. Ready for change.

This section focuses on the first point - how do we make our code easier to read? Here are some principles:

1. Use a linter/formatter.
2. Simple functions - every function should do one thing. This is the single responsibility principle.
3. Break up complex logic into multiple steps. In other words, prefer shorter lines instead of longer.
4. Do not do extended nesting. Instead of writing nested `if` clauses, prefer [`match`](https://docs.python.org/3/tutorial/controlflow.html#match-statements) or many `if` clauses on a single level.

You can automatically handle the first point - let's see how to install and use the `yapf` formatter extension in VS Code.

1. Open the `Extensions` tab, either by using the UI or by pressing `Ctrl + Shift + x`. You'll see somthing along the lines of:
  
![w01_yapf_on_vscode.png](./assets/w01_yapf_on_vscode.png "w01_yapf_on_vscode.png")

2. Search for `yapf`:

![w01_yapf_on_vscode_1.png](./assets/w01_yapf_on_vscode_1.png "w01_yapf_on_vscode_1.png")

3. Select and install it:

![w01_yapf_on_vscode_2.png](./assets/w01_yapf_on_vscode_2.png "w01_yapf_on_vscode_2.png")

4. After installing, please apply it on every Python file. To do so, press `F1` and type `Format Document`. The script would then be formatted accordingly.

![w01_yapf_on_vscode_3.png](./assets/w01_yapf_on_vscode_3.png "w01_yapf_on_vscode_3.png")

# Week 02 - Implementing Gradient Descent

!!!

- [ ] We created a chat in Messenger: DL_24-25!

!!!

## Backpropagation

<details>

<summary>How do we translate the expression "slope of a line"?</summary>

Наклон на линия.

</details>

<details>

<summary>How would you define the slope of a line?</summary>

- slope (also gradient) = a number that describes the direction of the line on a plane.
- often denoted by the letter $m$.

![w02_slope.png](assets/w02_slope.png "w02_slope.png")

- calculated as the ratio of the vertical change to the horizontal change ("rise over run") between two distinct points on the line:
  - a 45° rising line has slope $m = 1$ (tan(45°) = 1)
  - a 45° falling line has slope $m = -1$ (tan(-45°) = -1)

</details>

<details>

<summary>What is the sign of the slope of an increasing line going up from left to right?</summary>

Positive ($m > 0$).

</details>

<details>

<summary>What is the sign of the slope of a decreasing line going down from left to right?</summary>

Negative ($m < 0$).

</details>

<details>

<summary>What is the slope of a horizontal line?</summary>

$0$.

</details>

<details>

<summary>What is the slope of a vertical line?</summary>

A vertical line would lead to a $0$ in the denominator, so the slope can be regarder as `undefined` or `infinite`.

</details>

<details>

<summary>What is the steepness of a line?</summary>

- The absolute value of its slope:
  - greater absolute value indicates a steeper line.

</details>

<details>

<summary>Suppose a line runs through two points: P = (1, 2) and Q = (13, 8). What is its slope, direction and level of steepness?</summary>

$dy = 8 - 2 = 6$
$dx = 13 - 1 = 12$
$m = \frac{dy}{dx} = \frac{6}{12} = \frac{1}{2} = 0.5$

Direction: $0.5 > 0$ => up
Steepness: $0 < 0.5 < 1$ => not very steep (less steep than a 45° rising line)

</details>

<details>

<summary>Suppose a line runs through two points: P = (4, 15) and Q = (3, 21). What is its slope, direction and level of steepness?</summary>

$dy = 21 - 15 = 6$
$dx = 3 - 4 = -1$
$m = \frac{dy}{dx} = \frac{6}{-1} = -6$

Direction: $-6 < 0$ => down
Steepness: $|-6| = 6 > 1$ => steep

</details>

<details>

<summary>What is the link between "slope" and "derivative"?</summary>

- For non-linear functions, the rate of change varies along the curve.
- The derivative of the function at a point
$=$ The slope of the line, tangent to the curve at the point
$=$ The rate of change of the function at that point

![w02_slop_der_connection.png](assets/w02_slop_der_connection.png "w02_slop_der_connection.png")

Formula for slope:

$m = \frac{dy}{dx}$

Formula for derivative:

${\displaystyle L=\lim _{eps\to 0}{\frac {f(x+eps)-f(x)}{eps}}}$

it's the same formula as for the slope, only here the change in $x$ is infinitesimally small.

For example, let $f$ be the squaring function: ${\displaystyle f(x)=x^{2}}$. Then the derivative is:

$$\frac{f(x+eps) - f(x)}{eps} = \frac{(x+eps)^2 - x^2}{eps} = \frac{x^2 + 2xeps + eps^2 - x^2}{eps} = 2x + eps$$

The division in the last step is valid as long as $eps \neq 0$. The closer $eps$ is to $0$, the closer this expression becomes to the value $2x$. The limit exists, and for every input $x$ the limit is $2x$. So, the derivative of the squaring function is the doubling function: ${\displaystyle f'(x)=2x}$.

</details>

<details>

<summary>So, what added value does the derivative have?</summary>

**It tells us by how much the value of a function increases when we *increase* its input by a tiny bit.**

Do we remember the below diagram?

![w01_multiplier_loss_viz.png](assets/w01_multiplier_loss_viz.png "w01_multiplier_loss_viz.png")

</details>

<details>

<summary>What are the rules of derivatives that you can recall - write out the rule and an example of it?</summary>

Recall the rules of computation [here](https://en.wikipedia.org/wiki/Derivative#Rules_of_computation).

Also, recall the chain rule [here](https://en.wikipedia.org/wiki/Chain_rule).

<details>

<summary>What is the derivative of sin(6x)?</summary>

$\frac{d}{dx}[\sin(6x)] = \cos(6x) * \frac{d}{dx}[6x] = \cos(6x) * 6 = 6\cos(6x)$

See how the above corresponds with this definition:

$${\displaystyle {\frac {dz}{dx}}={\frac {dz}{dy}}\cdot {\frac {dy}{dx}},}$$

$z = \sin$
$y = 6x$

In other words, $x$ influences the value of $\sin$ through the value of $y=6x$.

</details>

</details>

<details>

<summary>What is backpropagation then?</summary>

Backpropagation is the iterative process of calculating derivatives of the loss function with respect to every `value` node leading up to it.

Rules of thumb:

```text
Start from the final child (the last node in topological order).
+ => copy gradient to parents:
    parent1.grad = current.grad
    parent2.grad = current.grad
* => multiply value of other parent with current gradient:
    parent1.grad = parent2.value * current.grad
    parent2.grad = parent1.value * current.grad
```

Let's say we have the following computational graph and we have to see how tiny changes in the weights and biases influence the value of `L`:

![w02_03_result](assets/w02_03_result.svg?raw=true "w02_03_result.png")

<details>

<summary>Reveal answer</summary>

![w02_calculations](assets/w02_calculations.png "w02_calculations.png")

End:

![w02_04_result](assets/w02_04_result.svg?raw=true "w02_04_result.png")

</details>

</details>

## Topological sort

Topological ordering of a directed graph is a linear ordering of its vertices such that for every directed edge $(u,v)$ from vertex $u$ to vertex $v$, $u$ comes before $v$ in the ordering.

The canonical application of topological sorting is in scheduling a sequence of jobs or tasks based on their dependencies.

Two ways to sort elements in topological order are given in [Wikipedia](https://en.wikipedia.org/wiki/Topological_sorting).

## The hyperbolic tangent

<details>

<summary>Why are activation functions needed?</summary>

They introduce nonlinearity, making it possible for our network to learn non-linear transformations. Composition of matrices is a single matrix (as the matrix is a linear operation).

</details>

$${\displaystyle \tanh x={\frac {\sinh x}{\cosh x}}={\frac {e^{x}-e^{-x}}{e^{x}+e^{-x}}}={\frac {e^{2x}-1}{e^{2x}+1}}.}$$

We observe that the `tanh` function is a shifted and stretched version of the `sigmoid`. Below, we can see its plot when the input is in the range $[-10, 10]$:

![w02_tanh](assets/w02_tanh.png "w02_tanh.png")

The output range of the tanh function is $(-1, 1)$ and presents a similar behavior with the `sigmoid` function. Thus, the main difference is the fact that the `tanh` function pushes the input values to $1$ and $-1$ instead of $1$ and $0$.

The important difference between the two functions is the behavior of their gradient.

$${\frac {d}{dx}}\sigma(x) = \sigma(x) (1 - \sigma(x))$$
$${\frac {d}{dx}}\tanh(x) = 1 - \tanh^{2}(x)$$

![w02_tanh_sigmoid_gradients](assets/w02_tanh_sigmoid_gradients.png "w02_tanh_sigmoid_gradients.png")

Using the `tanh` activation function results in higher gradient values during training and higher updates in the weights of the network. So, if we want strong gradients and big steps, we should use the `tanh` activation function.

Another difference is that the output of `tanh` is symmetric around zero, which could sometimes lead to faster convergence.

## Python OOP (Magic Methods)

### Initialization and Construction

- `__init__`: To get called by the `__new__` method. This is the `constructor` function for Python classes.
- `__new__`: To get called in an object’s instantiation (**do not use unless no other option**).
- `__del__`: It is the destructor (**do not use unless no other option**).

### Arithmetic operators

- `__add__(self, other)`: Implements behavior for the `+` operator (addition).
- `__sub__(self, other)`: Implements behavior for the `–` operator (subtraction).
- `__mul__(self, other)`: Implements behavior for the `*` operator (multiplication).
- `__floordiv__(self, other)`: Implements behavior for the `//` operator (floor division).
- `__truediv__(self, other)`: Implements behavior for the `/` operator (true division).
- `__mod__(self, other)`: Implements behavior for the `%` operator (modulus).
- `__pow__(self, other)`: Implements behavior for the `**` operator (exponentiation).
- `__and__(self, other)`: Implements behavior for the `&` operator (bitwise and).
- `__or__(self, other)`: Implements behavior for the `|` operator (bitwise or).
- `__xor__(self, other)`: Implements behavior for the `^` operator (bitwise xor).
- `__neg__(self)`: Implements behavior for negation using the `–` operator.

### String Magic Methods

- `__str__(self)`: Defines behavior for when `str()` is called on an instance of your class.
- `__repr__(self)`: To get called by built-int `repr()` method to return a machine readable representation of a type. **This method gets called when an object is passed to the `print` function.**

### Comparison magic methods

- `__eq__(self, other)`: Defines behavior for the equality operator, `==`.
- `__ne__(self, other)`: Defines behavior for the inequality operator, `!=`.
- `__lt__(self, other)`: Defines behavior for the less-than operator, `<`.
- `__gt__(self, other)`: Defines behavior for the greater-than operator, `>`.
- `__le__(self, other)`: Defines behavior for the less-than-or-equal-to operator, `<=`.
- `__ge__(self, other)`: Defines behavior for the greater-than-or-equal-to operator, `>=`.

# Week 03 - Hello, PyTorch

## PyTorch. A deep learning framework

<details>

<summary>Having done everything until now, what's the biggest differentiating factor between machine learning algorithms and deep learning algorithms?</summary>

Machine learning relies on **hand-crafted** feature engineering.

Deep learning enable **automatic** feature engineering from raw data. Automatic feature engineering is also known as **representation learning**.

</details>

<details>

<summary>Have you used PyTorch before? What problems did it help you solve?</summary>

PyTorch:

- is one of the most popular deep learning frameworks;
- is the framework used in many published deep learning papers;
- is intuitive and user-friendly;
- has much in common with NumPy;
- has a really good documentation. Check it out [here](https://pytorch.org/).

Be sure to update your virtual environment (if you haven't done so already): `pip install -Ur requirements.txt`.

</details>

## Tensors. The building blocks of networks

### What is a tensor?

A wrapper around an **n-dimensional NumPy array**, i.e. a class that has extended functionality (e.g. automatic backpropagation).

![w03_tensor_meme](assets/w03_tensor_meme.png "w03_tensor_meme.png")

### Creating tensors

**From a Python list:**

```python
import torch
xs = [[1, 2, 3], [4, 5, 6]]
tensor = torch.tensor(xs)
tensor
```

```console
tensor([[1, 2, 3],
        [4, 5, 6]])
```

```python
type(tensor)
```

```console
<class 'torch.Tensor'>
```

**From a NumPy array:**

```python
import numpy as np
import torch
np_array = np.array([1, 2, 3])
np_tensor = torch.from_numpy(np_array)
np_tensor
```

```console
tensor([1, 2, 3])
```

**Creating a tensor via `torch.tensor` vs `torch.Tensor`. When to choose which?**

- `torch.tensor` infers the data type automatically.
- `torch.Tensor` returns a `FloatTensor`.
- Advice: Stick to `torch.tensor`.

### Useful attributes

```python
import torch
xs = [[1, 2, 3], [4, 5, 6]]
tensor = torch.tensor(xs)
tensor.shape, tensor.dtype
```

```console
(torch.Size([2, 3]), torch.int64)
```

- Deep learning often requires a GPU, which, compared to a CPU can offer:
  - parallel computing capabilities;
  - faster training times.
- To see on which device the Tensor is currently sitting it, we can use the `.device` attribute:

```python
tensor.device
```

```console
device(type='cpu')
```

### Shapes matter

**Compatible:**

```python
a = torch.tensor([
    [1, 1],
    [2, 2],
])
b = torch.tensor([
    [2, 2],
    [3, 3],
])

a + b
```

```console
tensor([[3, 3],
        [5, 5]])
```

**Incompatible:**

```python
a = torch.tensor([
    [1, 1],
    [2, 2],
])
b = torch.tensor([
    [2, 2, 4],
    [3, 3, 4],
])

a + b
```

```console
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1
```

### Multiplication

<details>

<summary>What is broadcasting?</summary>

An implicit operation that copies an element (or a group of elements) `n` times along a dimension.

</details>

Element-wise multiplication can be done with the operator `*`:

```python
a = torch.tensor([
    [1, 1],
    [2, 2],
])
b = torch.tensor([
    [2, 2],
    [3, 3],
])

a * b
```

```console
tensor([[2, 2],
        [6, 6]])
```

We can do matrix multiplication with the function `torch.matmul`:

```python
# vector x vector
tensor1 = torch.randn(3)
tensor2 = torch.randn(3)
res = torch.matmul(tensor1, tensor2)
res, res.size()

# matrix x vector
tensor1 = torch.randn(3, 4)
tensor2 = torch.randn(4)
torch.matmul(tensor1, tensor2).size()

# batched matrix x broadcasted vector
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4)
torch.matmul(tensor1, tensor2).size()

# batched matrix x batched matrix
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(10, 4, 5)
torch.matmul(tensor1, tensor2).size()

# batched matrix x broadcasted matrix
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4, 5)
torch.matmul(tensor1, tensor2).size()
```

```console
(tensor(0.7871), torch.Size([]))
torch.Size([3])
torch.Size([10, 3])
torch.Size([10, 3, 5])
torch.Size([10, 3, 5])
```

Check other built-in functions [here](https://pytorch.org/docs/main/torch.html).

## Our first neural network using PyTorch

We'll begin by building a basic, two-layer network with no hidden layers.

![w03_first_nn.png](assets/w03_first_nn.png "w03_first_nn.png")

All functions and classes related to creating and managing neural networks can be explored in the [`torch.nn` module](https://pytorch.org/docs/stable/nn.html).

```python
import torch
import torch.nn as nn

linear_layer = nn.Linear(in_features=3, out_features=2)

user_data_tensor = torch.tensor([[0.3471, 0.4547, -0.2356]])
output = linear_layer(user_data_tensor)
output
```

```console
tensor([[-0.7252,  0.3228]], grad_fn=<AddmmBackward0>)
```

[Linear layer](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#linear):

- `in_features` (`int`) – size of each input sample;
- `out_features` (`int`) – size of each output sample;
- `bias` (`bool`) – If set to `False`, the layer will not learn an additive bias. Default: `True`.
- each linear layer has a `.weight` attribute:

```python
linear_layer.weight
```

```console
Parameter containing:
tensor([[-0.1971, -0.4996,  0.1233],
        [ 0.2203,  0.3508, -0.1402]], requires_grad=True)
```

- and a `.bias` attribute (by default):

```python
linear_layer.bias
```

```console
Parameter containing:
tensor([-0.4006,  0.0538], requires_grad=True)
```

For input $X$, weights $W_0$ and bias $b_0$, the linear layers performs:

$$y_0 = W_0 \cdot X + b_0$$

![w03_linear_op.png](assets/w03_linear_op.png "w03_linear_op.png")

- in the example above, the linear layer is used to transform the output from shape $(1, 3)$ to shape $(1, 2)$. We refer to $1$ as the **batch size**: how many observations were passed at once to the neural network.
- networks with only linear layers are called **fully connected**: each neuron in a layer is connected to each neuron in the next layer.

## Stacking layers with `nn.Sequential()`

We can easily compose multiple layers using the [`Sequential` class](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html#sequential):

```python
model = nn.Sequential(
    nn.Linear(10, 18),
    nn.Linear(18, 20),
    nn.Linear(20, 5),
)
model
```

```console
Sequential(
  (0): Linear(in_features=10, out_features=18, bias=True)
  (1): Linear(in_features=18, out_features=20, bias=True)
  (2): Linear(in_features=20, out_features=5, bias=True)
)
```

- Input is passed through the linear layers automatically.
- Here's how the sizes change: **Input 10** => output 18 => output 20 => **Output 5**.

```python
input_tensor
```

```console
tensor([[-0.0014,  0.4038,  1.0305,  0.7521,  0.7489, -0.3968,  0.0113, -1.3844,
          0.8705, -0.9743]])
```

```python
output_tensor = model(input_tensor)
output_tensor
```

```console
tensor([[-0.2361, -0.0336, -0.3614,  0.1190,  0.0112]],
       grad_fn=<AddmmBackward0>)
```

The output of this neural network:

- is still not yet meaningful. This is because the weights and biases are initially random floating-point values.
- is called a **logit**: non-normalized, raw network output.

## Checkpoint

What order should the following blocks be in, in order for the snippet to be correct:

1. `nn.Sequential(`
2. `nn.Linear(14, 3)`
3. `)`
4. `nn.Linear(3, 2)`
5. `nn.Linear(20, 14)`
6. `nn.Linear(5, 20)`

<details>

<summary>Reveal answer</summary>

1, 6, 5, 2, 4, 3

</details>

## Stacked linear transformations is still just one big linear transformation

Applying multiple stacked linear layers is equivalent to applying one linear layer that's their composition [[proof](https://www.3blue1brown.com/lessons/matrix-multiplication)]:

![w03_matrix_composition.png](assets/w03_matrix_composition.png "w03_matrix_composition.png")

<details>

<summary>What is the problem with this approach?</summary>

We still have a model that can represent **only linear relationships** with the input.

![w03_stacked_lin_layers.png](assets/w03_stacked_lin_layers.png "w03_stacked_lin_layers.png")

</details>

<details>

<summary>How do we fix this?</summary>

We need to include a transformation (i.e. a function) that is non-linear, so that we can also model nonlinearities. Such functions are called **activation functions**.

![w03_activation_fn.png](assets/w03_activation_fn.png "w03_activation_fn.png")

- **Activation functions** add **non-linearity** to the network.
- A model can learn more **complex** relationships with non-linearity.
- `Logits` are passed to the activation functions and the results are called `activations`.

</details>

<details>

<summary>What activations functions have you heard of?</summary>

- `sigmoid`: Binary classification.

![w03_sigmoid_graph.png](assets/w03_sigmoid_graph.png "w03_sigmoid_graph.png")

- `softmax`: Softmax for multi-class classification.

![w03_sigmoid_softmax.png](assets/w03_sigmoid_softmax.png "w03_sigmoid_softmax.png")

- `relu`.
- `tanh`.
- etc, etc.

![w03_activation_fns_grpahs.png](assets/w03_activation_fns_grpahs.png "w03_activation_fns_grpahs.png")
</details>

## Sigmoid in PyTorch

### Individually

**Binary classification** task:

- Predict whether an animal is **1 (mammal)** or **0 (not mammal)**.
- We take the logit: `6`.
- Pass it to the sigmoid.
- Obtain a value between `0` and `1` and treat it as a probability.

![w03_sigmoid.png](assets/w03_sigmoid.png "w03_sigmoid.png")

```python
input_tensor = torch.tensor([[6.0]])
sigmoid = nn.Sigmoid()
probability = sigmoid(input_tensor)
probability
```

```console
tensor([[0.9975]])
```

### As part of a network

Sigmoid as a last step in a network of stacked linear layers is equivalent to traditional logistic regression.

```python
model = nn.Sequential(
    nn.Linear(6, 4),
    nn.Linear(4, 1),
    nn.Sigmoid(),
)
```

## Softmax

- Takes N-element vector as input and outputs vector of same size.
- Outputs a probability distribution:
  - each element is a probability (it's bounded between `0` and `1`).
  - the sum of the output vector is equal to `1`.

![w03_softmax_nn.png](assets/w03_softmax_nn.png "w03_softmax_nn.png")

- `dim=-1` indicates softmax is applied to the input tensor's last dimension:

```python
input_tensor = torch.tensor([[4.3, 6.1, 2.3]])
softmax = nn.Softmax(dim=-1)
probabilities = softmax(input_tensor)
probabilities
```

```console
tensor([[0.1392, 0.8420, 0.0188]])
```

## Checkpoint

Which of the following statements about neural networks are true? (multiple selection)

A. A neural network with a single linear layer followed by a sigmoid activation is similar to a logistic regression model.
B. A neural network can only contain two linear layers.
C. The softmax function is widely used for multi-class classification problems.
D. The input dimension of a linear layer must be equal to the output dimension of the previous layer.

<details>

<summary>Reveal answer</summary>

A, C, D.

</details>

## Training a network

<details>

<summary>List out the steps of the so-called forward pass.</summary>

1. Input data is passed forward or propagated through a network.
2. Computations are performed at each layer.
3. Outputs of each layer are passed to each subsequent layer.
4. Output of the final layer is the prediction(s).

</details>

<details>

<summary>List the steps that are performed in the "training loop"?</summary>

1. Forward pass.
2. Compare outputs to true values.
3. Backpropagate to update model weights and biases.
4. Repeat until weights and biases are tuned to produce useful outputs.

</details>

<details>

<summary>Wait - why was the loss function needed again?</summary>

It dictates how the weights and biases should be tweaked to more closely resemble the training distribution of labels.

Ok - let's say that:

- $y$ is a single integer (class label), e.g. $y = 0$;
- $\hat{y}$ is a tensor (output of softmax), e.g. $[0.57492, 0.034961, 0.15669]$.

<details>

<summary>How do we compare an integer to a tensor when the task is classification?</summary>

We one-hot encode the integer and pass both of them to the cross entropy loss function.

![w03_ohe.png](assets/w03_ohe.png "w03_ohe.png")

OHE in Pytorch:

```python
import torch.nn.functional as F
F.one_hot(torch.tensor(0), num_classes=3)
```

```console
tensor([1, 0, 0])
```

Cross entropy (multiclass):

$H(p,q)\ =\ -\sum _{i}p_{i}\log q_{i}$

Binary case:

$H(p,q)\ =\ -\sum _{i}p_{i}\log q_{i}\ =\ -y\log {\hat {y}}-(1-y)\log(1-{\hat {y}})$

</details>

</details>

## Cross-entropy loss in PyTorch

The PyTorch implementation has built-in softmax, so it takes logits and target classes.

```python
import torch
from torch.nn import CrossEntropyLoss

scores = torch.tensor([[-0.1211,  0.1059]])
one_hot_target = torch.tensor([[1., 0.]])

criterion = CrossEntropyLoss()
criterion(scores, one_hot_target)
```

```console
tensor(0.8131)
```

## Minimizing the loss

![w03_derivative_valley.png](assets/w03_derivative_valley.png "w03_derivative_valley.png")

### Backpropagation

Consider a network made of three layers, $L0$, $L1$ and $L2$:

- we calculate local gradients for $L0$, $L1$ and $L2$ using backpropagation;
- we calculate loss gradients with respect to $L2$, then use $L2$ gradients to calculate $L1$ gradients and so on.

![w03_backprop.png](assets/w03_backprop.png "w03_backprop.png")

PyTorch does automatic backpropagation:

```python
criterion = CrossEntropyLoss()
loss = criterion(prediction, target)
loss.backward() # compute the gradients
```

```python
# Access each layer's gradients
model[0].weight.grad, model[0].bias.grad
model[1].weight.grad, model[1].bias.grad
model[2].weight.grad, model[2].bias.grad
```

We can then update the model paramters.

Here's how this can be done manually:

```python
lr = 0.001

weight = model[0].weight
weight_grad = model[0].weight.grad
weight = weight - lr * weight_grad

bias = model[0].bias
bias_grad = model[0].bias.grad
bias = bias - lr * bias_grad
```

### Optimizers

In PyTorch, an **optimizer** takes care of weight updates. Different optimizers have different logic for updating model parameters (or weights) after calculation of local gradients. Some built-in optimizers include:

- RMSProp;
- Adam;
- AdamW;
- SGD (Stochastic Gradient Descent).

They are all used in the same manner:

```python
from torch import 

optimizer = optim.SGD(model.parameters(), lr=0.001)

<... trainig loop ...>

optimizer.zero_grad() # make the current gradients 0
loss.backward()       # calculate the new gradients
optimizer.step()      # update the parameters

<... trainig loop ...>
```

## Putting it all together. Training a neural network

<details>

<summary>List the steps that would be used to train a neural network.</summary>

1. Create a dataset.
2. Create a model.
3. Define a loss function.
4. Define an optimizer.
5. Run a training loop, where for each batch of samples in the dataset, we repeat:
   1. Zeroing the graidents.
   2. Forward pass to get predictions for the current training batch.
   3. Calculating the loss.
   4. Calculating gradients.
   5. Updating model parameters.

</details>

<details>

<summary>What metrics can we use for regressions problems?</summary>

The mean squared error loss:

$$MSE = \frac{1}{N} * \sum(y - \hat{y})^2$$

In PyTorch:

```python
criterion = nn.MSELoss()
loss = criterion(prediction, target)
print(loss.item())
```

</details>

## Creating dataset and dataloader

- `TensorDataset`: acts as a wrapper around our features and targets.
- `DataLoader`: splits the dataset into batches.

```python
dataset = TensorDataset(torch.tensor(features).float(), torch.tensor(target).float()) # has to be the same datatype as the parameters of the model
input_sample, label_sample = dataset[0]

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
for batch_inputs, batch_labels in dataloader:
    print(f'{batch_inputs=}')
    print(f'{batch_labels=}')
```

## Gradients of the sigmoid and softmax functions

![w03_sigmoid_gradient.png](assets/w03_sigmoid_gradient.png "w03_sigmoid_gradient.png")

Gradients:

- Approach $0$ for low and high values of $x$.
- The far left (value $0$) and far right (value $1$) regions are known as **saturation regions** because the gradient/derivative there is too small, slowing down learning.
  - Learning slows down when the gradient is small, because the weight upgrade of the network at each iteration is directly proportional to the gradient magnitude.
  - The sooner the learning starts to slow down, the less the first layers are going to learn. This is known as **the vanishing gradient problem**.

## Introducing the **Re**ctified **L**inear **U**nit (`ReLU`)

![w03_relu_gradient.png](assets/w03_relu_gradient.png "w03_relu_gradient.png")

<details>

<summary>Looking at the graph, what is the function that ReLU applies?</summary>

$f(x) = max(x, 0)$

In PyTorch:

```python
relu = nn.ReLU()
```

</details>

<details>

<summary>What is the output for positive input?</summary>

The output is equal to the input.

</details>

<details>

<summary>What is the output for negative inputs?</summary>

$0$.

</details>

<details>

<summary>Why does ReLU solve the vanishing gradient problem?</summary>

Because it has a deriviative value of $1$ for large positive values as well.

</details>

<details>

<summary>However, what problem does ReLU introduce that is not present when using sigmoid?</summary>

The dying neuron problem.

A large gradient flowing through a ReLU neuron could cause the bias to update in such a way that it becomes very negative, which in turn leads to the neuron outputting only negative values.

This would mean that the derivative will be $0$ and in the future the weights and bias will not be updated.

It acts like permanent brain damage.

Note:

- In practice, dead ReLUs connections are not a **major** issue.
- Most deep learning networks can still learn an adequate representations with only sub-selection of possible connections.
  - This is possible because deep learning networks are highly over-parameterized.
- The computational effectiveness and efficiency of ReLUs still make them one of the best options currently available (even with the possible drawbacks of dead neurons).

</details>

<details>

<summary>How can we solve this?</summary>

## Introducing Leaky ReLU

![w03_leakyrelu_gradient.png](assets/w03_leakyrelu_gradient.png "w03_leakyrelu_gradient.png")

- Same behavior for positive inputs.
- Negative inputs get multiplied by a small coefficient: `negative_slope` (defaulted to $0.01$).
- The gradients for negative inputs are very small, but never $0$.

In PyTorch:

```python
leaky_relu = nn.LeakyReLU(negative_slope=0.05)
```

</details>

## Counting the number of parameters

### Layer naming conventions

![w03_layer_names.png](assets/w03_layer_names.png "w03_layer_names.png")

<details>

<summary>What is the dependency between the number of neurons in the input layer and the user data?</summary>

The number of neurons in the input layer depends on the number of features in a single observation.

</details>

<details>

<summary>What is the dependency between the number of neurons in the output layer and the user data?</summary>

The number of neurons in the output layer depends on the number of classes that can be assigned.

<details>

<summary>What if it's a regression problem?</summary>

Then the output layer is a single neuron.

So, we get the following architecture:

```python
model = nn.Sequential(nn.Linear(n_features, 8),
                      nn.Linear(8, 4),
                      nn.Linear(4, n_classes))
```

</details>

</details>

### PyTorch's `numel` method

- We could vary the number of neurons in the hidden layers (and the amount of hidden layers).
- However, we should remember that increasing the number of hidden layers = increasing the number of parameters = increasing the **model capacity**.

Given the followin model:

```python
n_features = 8
n_classes = 2

model = nn.Sequential(nn.Linear(n_features, 4),
                      nn.Linear(4, n_classes))
```

We can manually count the number of parameters:

- first layer has $4$ neurons, each connected to the $8$ neurons in the input layer and $1$ bias $= 36$ parameters.
- second layer has $2$ neurons, each connected to the $4$ neurons in the input layer and $1$ bias $= 10$ parameters.
- Total: $46$ learnable parameters.

In PyTorch, we can use the `numel` method to get the number of parameters of a neuron:

```python
total = 0
for parameter in model.parameters():
    total += parameter.numel()
print(total)
```

### Checkpoint

Calculate manually the number of parameters of the model below. How many does it have?

```python
model = nn.Sequential(nn.Linear(16, 4),
                      nn.Linear(4, 2),
                      nn.Linear(2, 1))
```

<details>

<summary>Reveal answer</summary>

$81$.

We can confirm it:

```python
model = nn.Sequential(nn.Linear(16, 4),
                      nn.Linear(4, 2),
                      nn.Linear(2, 1))

print(sum(param.numel() for param in model.parameters()))
```

</details>

## Learning rate and momentum

- Training a neural network = solving an **optimization problem**.
- Most algorithms have two parameters:
  - **learning rate**: controls the step size.
  - **momentum**: controls the inertia of the optimizer.
- Poor choice of values can lead to:
  - long training times.
  - bad overall performance.

### Optimal learning rate

Optimal learning rates vary between `1` and `0.0001`:

![w03_optimal_lr.png](assets/w03_optimal_lr.png "w03_optimal_lr.png")

Small learning rate = more training time:

![w03_small_lr.png](assets/w03_small_lr.png "w03_small_lr.png")

Big learning rate = oscilation:

![w03_big_lr.png](assets/w03_big_lr.png "w03_big_lr.png")

### Optimal momentum

Momentum is the functionality of an optimizer to continue making large steps when the previous steps were large. Thus, steps become small when we're in a valley.

![w03_momentum_formula.png](assets/w03_momentum_formula.png "w03_momentum_formula.png")

Reference taken from [this](https://arxiv.org/pdf/1609.04747) paper.

No momentum (`momentum=0`) = stuck in local minimum:

![w03_no_momentum.png](assets/w03_no_momentum.png "w03_no_momentum.png")

Optimal momentum values vary between `0.5` and `0.99`:

![w03_with_momentum.png](assets/w03_with_momentum.png "w03_with_momentum.png")

## Layer initialization

Often it can happen that the initial loss of the network is very high and then rather fast it decreases to a valley type of graph:

![w03_high_loss.png](assets/w03_high_loss.png "w03_high_loss.png")

This happens because the logits that come out in the very first iteration are:

- very high or very low numbers;
- the difference between each of them is very high, meaning that they are not very close to one another in terms of value.

We can solve this by making the logits closer together:

- be it around `0`;
- or just making them equal to each other.

This is because the softmax would then treat them as probabilities and the more "clustered" their values are, the more uniform the output distribution will be.

We can solve this by normalizing the weight's values.

Instead of writing this:

```python
import torch.nn as nn
layer = nn.Linear(64, 128)
print(layer.weight.min(), layer.weight.max())
```

```console
tensor(-0.1250, grad_fn=<MinBackward1>) tensor(0.1250, grad_fn=<MaxBackward1>)
```

We can write this:

```python
import torch.nn as nn
layer = nn.Linear(64, 128)
nn.init.uniform_(layer.weight)
print(layer.weight.min(), layer.weight.max())
```

```console
tensor(3.0339e-05, grad_fn=<MinBackward1>) tensor(1.0000, grad_fn=<MaxBackward1>)
```

## Transfer learning

### The goal

<details>

<summary>What have you heard about transfer learning?</summary>

Reusing a model trained on task for accomplishing a second similar task.

</details>

<details>

<summary>What is the added value?</summary>

- Faster training (fewer epochs).
- Don't need as large amount of data as would be needed otherwise.
- Don't need as many resources as would be needed otherwise.

</details>

<details>

<summary>Can we think of some examples?</summary>

We trained a model on a dataset of data scientist salaries in the US and want to get a new model on a smaller dataset of salaries in Europe.

</details>

### Fine-tuning

- A way to do transfer learning.
- Smaller learning rate.
- Not every layer is trained (some of the layers are kept **frozen**).

<details>

<summary>What does it mean to freeze a layer?</summary>

No updates are done to them (gradient for them is $0$).

</details>

<details>

<summary>Which layers should be frozen?</summary>

The early ones. The goal is to use (and change) the layers closer to the output layer.

In PyTorch:

```python
import torch.nn as nn

model = nn.Sequential(nn.Linear(64, 128),
                      nn.Linear(128, 256))

for name, param in model.named_parameters():
    if name == '0.weight':
        param.requires_grad = False
```

</details>

### Checkpoint

Order the sentences to follow the fine-tuning process.

1. Train with a smaller learning rate.
2. Freeze (or not) some of the layers in the model.
3. Load pre-trained weights.
4. Find a model trained on a similar task.
5. Look at the loss values and see if the learning rate needs to be adjusted.

<details>

<summary>Reveal answer</summary>

4, 3, 2, 1, 5

</details>

## The Water Potability Dataset

- Task: classify a water sample as potable or drinkable (`1` or `0`) based on its chemical characteristics.
- All features have been normalized to between zero and one. Two files are present in our `DATA` folder: `water_train.csv` and `water_test.csv`. Here's how both of them look like:

![w03_water_potability_datasets.png](assets/w03_water_potability_datasets.png "w03_water_potability_datasets.png")

## Evaluating a model on a classification task

Let's recall the steps for training a neural network:

1. Create a dataset.
2. Create a model.
3. Define a loss function.
4. Define an optimizer.
5. Run a training loop, where for each batch of samples in the dataset, we repeat:
   1. Zeroing the graidents.
   2. Forward pass to get predictions for the current training batch.
   3. Calculating the loss.
   4. Calculating gradients.
   5. Updating model parameters.

<details>

<summary>Could we elaborate a bit more on point 5 - what dataset are we talking about?</summary>

When training neural networks in a supervised fashion we typically break down all the labeled data we have into three sets:

| Name       | Percent of data | Description                                                                                                 |
| ---------- | --------------- | ----------------------------------------------------------------------------------------------------------- |
| Train      | 70-90           | Learn optimal values for model parameters                                                                   |
| Validation | 5-15            | Hyperparameter tuning (batch size, learning rate, number of layers, number of neurons, type of layers, etc) |
| Test       | 5-15            | Only used once to calculate final performance metrics                                                       |

</details>

<details>

<summary>What classification metrics have you heard of?</summary>

- Accuracy: percentage of correctly classified examples.
- Recall: from all true examples, what percentage did our model find.
- Precision: from all the examples are model labelled as true, what percentage of the examples are actually true.
- F1: harmonic mean of precision and recall.

![w03_precision_recall.png](assets/w03_precision_recall.png "w03_precision_recall.png")

<details>

<summary>When should we use accuracy?</summary>

Only when all of the classes are perfectly balanced.

</details>

<details>

<summary>What metic should we use when we have an unbalanced target label?</summary>

F1-score.

</details>

</details>

## Calculating validation loss

After each training epoch we iterate over the validation set and calculate the average validation loss.

It's important to put the model in an evaluation mode, so no gradients get calculated and all layers are used as if they were processing user data.

```python
validation_loss = 0.0
model.eval()
with torch.no_grad():
    for sample, label in validation_loader:
        model(sample)
        loss = criterion(outputs, labels)
        validation_loss += loss.item()
epoch_validation_loss = validation_loss / len(validation_loader)
model.train()
```

## The Bias-Variance Tradeoff

<details>

<summary>What have you heard about it?</summary>

It can be used to determine whether a model has reached its best capabilities, is underfitting or is overfitting.

![w03_bias_variance.png](assets/w03_bias_variance.png "w03_bias_variance.png")

<details>

<summary>What is underfitting?</summary>

High training loss and high validation loss.

</details>

<details>

<summary>What is overfitting?</summary>

Low training loss and high validation loss.

</details>

</details>

## Calculating accracy with [`torchmetrics`](https://lightning.ai/docs/torchmetrics/stable//index.html)

The package `torchmetrics` provides implementations of popular classification and regression metrics:

- [accuracy](https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html#id4).
- [recall](https://lightning.ai/docs/torchmetrics/stable/classification/recall.html#id4).
- [precision](https://lightning.ai/docs/torchmetrics/stable/classification/precision.html#id4).
- [f1-score](https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html#f-1-score).
- [mean squared error](https://lightning.ai/docs/torchmetrics/stable/regression/mean_squared_error.html#mean-squared-error-mse).

```python
import torchmetrics

metric = torchmetrics.Accuracy(task='multiclass', num_classes=3)
for samples, labels in dataloader:
    outputs = model(samples)
    acc = metric(outputs, labels.argmax(dim=-1))
acc = metric.compute()
print(f'Accuracy on all data: {acc}')
metric.reset()
```

## Fighting overfitting

<details>

<summary>What is the result of overfitting?</summary>

The model does not generalize to unseen data.

</details>

<details>

<summary>What are the causes of overfitting?</summary>

| Problem                     | Solution                              |
| --------------------------- | ------------------------------------- |
| Model has too much capacity | Reduce model size / Add dropout       |
| Weights are too large       | Use weight decay                      |
| Dataset is not large enough | Get more data / Use data augmentation |

</details>

## Using a [`Dropout` layer](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#dropout)

- During training, randomly zeroes some of the elements of the input tensor with probability `p`.
  - This would mean that the neuron did not "fire" / did not get triggered.
- Add after the activation function.
- Behaves differently during training and evaluation/prediction:
  - we must remember to switch modes using `model.train()` and `model.eval()`.

```python
nn.Sequential(
    nn.Linear(n_features, n_classes),
    nn.ReLU(),
    nn.Dropout(p=0.8))
```

We can try it out:

```python
import numpy as np
import torch
from torch import nn

m = nn.Dropout(p=0.2)

inp = torch.randn(20, 16)
(m(inp).view(-1).numpy() == 0).mean()
```

```console
np.float64(0.1925)
```

## Weight decay

```python
optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

- Weight decay adds `L2` penalty to loss function to discourage large weights and biases.
  - Effectively, this is done by increasing the gradient with a `weight_decay` fraction of the weights:

```python
d_p = p.grad.data
if weight_decay != 0:
    d_p.add_(weight_decay, p.data)
```

- Optimizer's `weight_decay` parameter takes values between `0` and `1`.
  - Typically small value, e.g. `1e-3`.
- The higher the parameter, the stronger the regularization, thus the less likely the model is to overgit.
- More on the topic [in this post](https://discuss.pytorch.org/t/how-pytorch-implement-weight-decay/8436/3).

> **Note:** Using strong regularization, results in slower training times.

## Data augmentation

![w03_data_augmentation.png](assets/w03_data_augmentation.png "w03_data_augmentation.png")

- End result: Increased size and diversity of the training set.
- We'll discuss different pros and cons of this strategy in the upcoming weeks.

## Steps to maximize model performance

1. Overfit the training set (rarely possible to a full extend).
   - We ensure that the problem is solvable using deep learning.
   - We set a baseline to aim for with the validation set.
2. Reduce overfitting.
   - Improve performance on the validation set.
3. Fine-tune hyperparameters.

### Step 1: overfit the training set

If this is not possible to do with the full training set due to memory constraints, modify the training loop to overfit a `batch_size` of points (`batch_size=1` is also a possibility).

```python
features, labels = next(iter(trainloader))
for i in range(1e3):
    outputs = model(features)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

- Should reach accuracy (or your choice of metric) `1.0` and loss close (or equal) to `0.0`.
- This also helps with finding bugs in the code or in the data.
- Use the default value for the learning rate.
- Deliverables:
    1. Large enough model.
    2. Minimum training loss.

### Step 2: reduce overfitting

- Start to keep track of:
  - training loss;
  - training metric values;
  - validation loss;
  - validation metric values.
- Experiment with:
  - Dropout;
  - Data augmentation;
  - Weight decay;
  - Reducing the model capacity.
- Keep track of each hyperparamter.
- Deliverables:
  1. Maximum metric value on the validation set.
  2. Minimum loss on the validation set.
  3. Plots validating model performance.

![w03_plots.png](assets/w03_plots.png "w03_plots.png")

Be careful to not increase the training loss and reduce the training metric by too much (overfitting-reduction strategies often lead to this):

![w03_too_much_regularization.png](assets/w03_too_much_regularization.png "w03_too_much_regularization.png")

### Step 3: fine-tune hyperparameters

Grid search:

![w03_grid_search_example.png](assets/w03_grid_search_example.png "w03_grid_search_example.png")

Random search:

![w03_random_search_example.png](assets/w03_random_search_example.png "w03_random_search_example.png")

# Week 04 - Convolutional Neural Networks. Building multi-input and multi-output models

## Custom PyTorch Datasets

We can create a custom dataset for our water potability data by inheriting the PyTorch [Dataset class](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset). All customs `Dataset` classes must implement the following methods:

- `__init__`: to loads and saves the data in the state of the class. Typically accepts a CSV or an already loaded numpy matrix;
- `__len__`: returns the number of instaces in the saved data;
- `__getitem__`: returns the features and label for a single sample. Note: this method returns **a tuple**! The first element is an array of the features, the second is the label.

See an example [here](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files).

> **Note:** While it's not shown in the example, please don't forget to initialize the parent class as well by calling `super().__init__()`.

## Checkpoint

What is the correct way of iterating through the dataloader and passing the inputs to the model?

A. `for img, alpha, labels in dataloader_train: outputs = net(img, alpha)`
B. `for img, alpha, labels in dataloader_train: outputs = net(img)`
C. `for img, alpha in dataloader_train: outputs = net(img, alpha)`
D. `for img, alpha in dataloader_train: outputs = net(img)`

<details>
<summary>Reveal answer</summary>

A.

</details>

## Class-Based PyTorch Model

This time we'll use another syntax to define models: the `class-based` approach. It provides more flexibility the the functional style.

Here's an example of sequential model definition:

```python
import torch.nn as nn

net = nn.Sequential(
  nn.Linear(9, 16),
  nn.ReLU(),
  nn.Linear(16, 8),
  nn.ReLU(),
  nn.Linear(8, 1),
  nn.Sigmoid(),
)
```

Here's how it can be re-written using the `class-based` approach:

```python
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(9, 16)
    self.fc2 = nn.Linear(16, 8)
    self.fc3 = nn.Linear(8, 1)
  
  def forward(self, x):
    x = nn.functional.relu(self.fc1(x))
    x = nn.functional.relu(self.fc2(x))
    x = nn.functional.sigmoid(self.fc3(x))
    return x

net = Net()
```

As can be seen above, every model should define the following two methods:

- `__init__()`: defines the layers that are used in the `forward()` method;
- `forward()`: defines what happens to the model inputs once it receives them; this is where you pass inputs through pre-defined layers.

By convention `torch.nn.functional` gets imported with an alias `F`. That means that the above body of `forward` can be rewritten like:

```python
import torch.nn.functional as F

...

x = F.relu(self.fc1(x))
x = F.relu(self.fc2(x))
x = F.sigmoid(self.fc3(x))
```

- PyTorch has many famous deep learning models already built-in.
- For example, various vision models can be found in the [torchvision.models package](https://pytorch.org/vision/0.9/models.html).

## Unstable gradients

- **Vanishing gradients**: Gradients get smaller and smaller during backward pass.

![w04_vanishing_gradients.png](./assets/w04_vanishing_gradients.png "w04_vanishing_gradients.png")

- Results:
  - Earlier layers get smaller parameter updates;
  - Model does not learn.
  - Loss becomes constant.

- **Exploding gradients**: Gradients get larger and larger during backward pass.

![w04_exploding_gradients.png](./assets/w04_exploding_gradients.png "w04_exploding_gradients.png")

- Results:
  - Parameter updates are too large.
  - Loss becomes higher and higher.

### Solutions to unstable gradients

1. Proper weights initialization.
2. More appropriate activation functions.
3. Batch normalization.

#### Proper weights initialization

Good weight initialization ensures that the:

- Variance of layer inputs = variance of layer outputs;
- Variance of gradients is the same before and after a layer.

How to achieve this depends on the activation function:

- For ReLU and similar (sigmoid included), we can use [He/Kaiming initialization](https://paperswithcode.com/method/he-initialization).

```python
import torch.nn.init as init

init.kaiming_uniform_(layer.weight) # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_
```

#### Batch normalization

- Good choice of initial weights and activations doesn't prevent unstable gradients during training (only during initialization).
- Solution is to add another transformation after each layer - batch normalization:
  1. Standardizes the layer's outputs by subtracting the mean and diving by the standard deviation **in the batch dimension**.
  2. Scales and shifts the standardized outputs using learnable parameters.
- Result:
  - Model learns optimal distribution of inputs for each layer.
  - Faster loss decrease.
  - Helps against unstable gradients during training.
- Available as [`nn.BatchNorm1d`](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html).
  - **Note 1:** The number of features has to be equal to the number of output neurons of the previous layer.
  - **Note 2:** Done after applying layer and before the activation.

## The Clouds dataset

We will be working with a dataset containing pictures of various types of clouds.

![w04_task05.png](assets/w04_task05.png "w04_task05.png")

<details>

<summary>How can we load one of those images in Python?</summary>

We can use the [`pillow`](https://pypi.org/project/pillow/) package. It is imported with the name `PIL` and has a very handly [`Image.open` function](https://pillow.readthedocs.io/en/latest/handbook/tutorial.html#using-the-image-class).

</details>

<details>

<summary>Wait - what is an image again?</summary>

- The image is a matrix of pixels ("picture elements").
- Each pixel contains color information.

![w04_image_pixels.png](assets/w04_image_pixels.png "w04_image_pixels.png")

- Grayscale images: integer in the range $[0 - 255]$.
  - 30:

    ![w04_image_gray.png](assets/w04_image_gray.png "w04_image_gray.png")

- Color images: three/four integers, one for each color channel (**R**ed, **G**reen, **B**lue, sometimes also **A**lpha).
  - RGB = $(52, 171, 235)$:

    ![w04_image_blue.png](assets/w04_image_blue.png "w04_image_blue.png")

</details>

## Converting pixels to tensors and tensors to pixels

### [`ToTensor()`](https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html#totensor)

- Converts pixels to float tensors (PIL image => `torch.float`).
- Scales values to $[0.0, 1.0]$.

### [`PILToTensor()`](https://pytorch.org/vision/main/generated/torchvision.transforms.PILToTensor#piltotensor)

- Converts pixels to `8`-bit unsigned integers (PIL image => `torch.uint8`).
- Does not scale values: they stay in the interval $[0, 255]$.

### [`ToPILImage()`](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.ToPILImage)

- Converts a tensor or an numpy `ndarray` to PIL Image.
- Does not change values.

## Loading images with PyTorch

The easiest way to build a `Dataset` object when we have a classification task is with a predefined directory structure.

As we load the images, we could also apply preprocessing steps using `torchvision.transforms`.

```text
clouds_train
  - cumulus
    - 75cbf18.jpg
    - ...
  - cumulonimbus
  - ...
clouds_test
  - cumulus
  - cumulonimbus
```

- Main folders: `clouds_train` and `clouds_test`.
  - Inside: one folder per category.
    - Inside: image files.

```python
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils import data

train_transforms = transforms.Compose([
  transforms.ToTensor(), # convert the object into a tensor
  transforms.Resize((128, 128)), # resize the images to be of size 128x128
])

dataset_train = ImageFolder(
  'DATA/clouds/clouds_train',
  transform=train_transforms,
)

dataloader_train = data.DataLoader(
  dataset_train,
  shuffle=True,
  batch_size=1,
)

image, label = next(iter(dataloader_train))
print(image.shape)
```

```console
torch.Size([1, 3, 128, 128])
```

In the above output:

- `1`: batch size;
- `3`: three color channels;
- `128`: height;
- `128`: width.

We could display these images as well, but we'll have to do two transformations:

1. We need to have a three dimensional matrix. The above shape represents a `4D` one. To remove all dimensions with size `1`, we can use the `squeeze` method of the `image` object.
2. The number of color channels must come after the height and the width. To change the order of the dimensions, we can use the `permute` method of the `image` object.

```python
image = image.squeeze().permute(1, 2, 0)
print(image.shape)
```

```console
torch.Size([128, 128, 3])
```

We can now, plot this using `matplotlib`:

```python
import matplotlib.pyplot as plt
plt.imshow(image)
plt.axis('off')
plt.show()
```

![w04_loading_image_result.png](assets/w04_loading_image_result.png "w04_loading_image_result.png")

## Data augmentation

<details>

<summary>What is data augmentation?</summary>

Applying random transformations to original data points.

</details>

<details>

<summary>What is the goal of data augmentation?</summary>

Generating more data.

</details>

<details>

<summary>On which set should data augmentation be applied to - train, validation, test, all, some?</summary>

Only to the training set.

</details>

<details>

<summary>What is the added value of data augmentation?</summary>

- Increase the size of the training set.
- Increase the diversity of the training set.
- Improve model robustness.
- Reduce overfitting.

</details>

All supported image augmentation transformation can be found in [the documentation of torchvision](https://pytorch.org/vision/stable/transforms.html#v2-api-reference-recommended).

<details>

<summary>Image augmentation operations can sometimes negatively impact the training process. Can you think of two deep learning tasks in which specific image augmentation operations should not be used?</summary>

- Fruit classification and changing colors:

One of the supported image augmentation transformations is [`ColorJitter`](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_illustrations.html#colorjitter) - it randomly changes brightness, contrast, saturation, hue, and other properties of an image.

If we are doing fruit classification and decide to apply a color shift augmentation to an image of the lemon, the augmented image will still be labeled as lemon although it would represent a lime.

![w04_data_augmentation_problem.png](assets/w04_data_augmentation_problem.png "w04_data_augmentation_problem.png")

- Hand-written characters classification and vertical flip:

![w04_data_augmentation_problem2.png](assets/w04_data_augmentation_problem2.png "w04_data_augmentation_problem2.png")

</details>

<details>

<summary>So, how do we choose appropriate augmentation operations?</summary>

- Whether an augmentation operation is appropriate depends on the task and data.
- Remember: Augmentations impact model performance.

<details>

<summary>Ok, but then how do we see the dependence in terms of data?</summary>

Explore, explore, explore!

</details>

</details>

<details>

<summary>What transformations can you think of for our current task (cloud classification)?</summary>

```python
train_transforms = transforms.Compose([
  transforms.RandomHorizontalFlip(), # simulate different viewpoints of the sky
  transforms.RandomRotation(45), # expose model to different angles of cloud formations
  transforms.RandomAutocontrast(), # simulate different lighting conditions
  transforms.ToTensor(), # convert the object into a tensor
  transforms.Resize((128, 128)), # resize the images to be of size 128x128
])
```

![w04_data_augmentation.png](assets/w04_data_augmentation.png "w04_data_augmentation.png")

</details>

Which of the following statements correctly describe data augmentation? (multiple selection)

A. Using data augmentation allows the model to learn from more examples.
B. Using data augmentation increases the diversity of the training data.
C. Data augmentation makes the model more robust to variations and distortions commonly found in real-world images.
D. Data augmentation reduces the risk of overfitting as the model learns to ignore the random transformations.
E. Data augmentation introduces new information to the model that is not present in the original dataset, improving its learning capability.
F. None of the above.

<details>

<summary>Reveal answer</summary>

Answers: A, B, C, D.

Data augmentation allows the model to learn from more examples of larger diversity, making it robust to real-world distortions.

It tends to improve the model's performance, but it does not create more information than is already contained in the original images.

<details>

<summary>What should we prefer - using more real training data or generating it artificially?</summary>

If available, using more training data is preferred to creating it artificially with data augmentation.

</details>

</details>

## CNNs - The neural networks for image processing

Let's say that we have the following image:

![w04_linear_layers_problem.png](assets/w04_linear_layers_problem.png "w04_linear_layers_problem.png")

<details>

<summary>What is the problem of using linear layers to solve the classification task?</summary>

Too many parameters.

If the input size is `256x256`, that means that the network has `65,536` inputs!

If the first linear layer has `1000` neurons, only it alone would result in over `65` **million** parameters! For a color image, this number would be even higher.

![w04_linear_layers_problem2.png](assets/w04_linear_layers_problem2.png "w04_linear_layers_problem2.png")

So, the three main problems are:

- Incredible amount of resources needed.
- Slow training.
- Overfitting.

</details>

<details>

<summary>What is another more subtle problem of using linear layers only?</summary>

They are not space-invariant.

Linearly connected neurons could learn to detect the cat, but the same cat **won't be recognized if it appears in a *different* location**.

![w04_space_invariance.png](assets/w04_space_invariance.png "w04_space_invariance.png")

</details>

<details>

<summary>So, ok - the alternative is using CNNs. How do they work?</summary>

![w04_cnn.png](assets/w04_cnn.png "w04_cnn.png")

- Parameters are collected in one or more small grids called **filters**.
- **Slide** filter(s) over the input.
- At each step, perform the **convolution** operation.
- The end result (after the whole image has been traversed with **one filter**) is called a **feature map**:
  - Preserves spatial patterns from the input.
  - Uses fewer parameters than linear layer.
- Remember: one filter = one feature map. We can slide **multiple filters over an original image**, to get multiple feature maps.
- We can then apply activations to the feature maps.
- The set of all feature maps combined, form the output of a single convolutional layer.
- Available in `torch.nn`: [`nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html):
  - `in_channels`: number of input dimensions.
  - `out_channels`: number of filters.
  - `kernel_size`: height and width of the filter(s).

</details>

<details>

<summary>What does the convolution operation comprise of?</summary>

![w04_cnn_dot.png](assets/w04_cnn_dot.png "w04_cnn_dot.png")

</details>

<details>

<summary>What is zero padding?</summary>

![w04_cnn_zero_pad.png](assets/w04_cnn_zero_pad.png "w04_cnn_zero_pad.png")

- Add a frame of zeros to the input of the convolutional layer.
- This maintains the spatial dimensions of the input and output tensors.
- Ensures border pixels are treated equally to others.

Available as `padding` argument: `nn.Conv2d(3, 32, kernel_size=3, padding=1)`.

</details>

<details>

<summary>What is max pooling?</summary>

![w04_cnn_max_pool.png](assets/w04_cnn_max_pool.png "w04_cnn_max_pool.png")

- Slide non-overlapping window over input.
- At each position, retain only the maximum value.
- Used after convolutional layers to reduce spatial dimensions.

Available in `torch.nn`: [`nn.MaxPool2d(kernel_size=2)`](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html).

</details>

<details>

<summary>But what happens when the input is three dimensional - what do the filters look like?</summary>

The `2D` part of a `2D` convolution does not refer to the dimension of the convolution input, nor of dimension of the filter itself, but rather of the space in which the filter is allowed to move (`2` directions only).

Different `2D`-array filters are applied to each dimension and then their outputs are summed up (you can also think of this as a single `3D` matrix):

![w04_rgb_convolution.png](assets/w04_rgb_convolution.png "w04_rgb_convolution.png")

$$(1*1+2*2+4*3+5*4)+(0*0+1*1+3*2+4*3) = 56$$

More on this [here](https://stackoverflow.com/a/62544803/16956119) and [here](https://d2l.ai/chapter_convolutional-neural-networks/channels.html).

</details>

## Architecture

The typical architecture follows the style:

1. Convolution.
2. Activation function - `ReLU`, `ELU`, etc.
3. Max pooling.
4. Iterate the above until there are much more filters than heigh and width (effectively, until we get a much "deeper" `z`-axis).
5. Flatter everything into a single vector using [`nn.Flatten`](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html). This vector is the ***summary of the original input image***.
6. Apply several regular linear layers to it.

![w04_architecture2.png](assets/w04_architecture2.png "w04_architecture2.png")

Here's one famous architecture - [VGG-16](https://arxiv.org/abs/1409.1556v6):

![w04_architecture.png](assets/w04_architecture.png "w04_architecture.png")

In our case, we could have something like the following:

![w04_architecture_ours.png](assets/w04_architecture_ours.png "w04_architecture_ours.png")

Which of the following statements are true about convolutional layers? (multiple selection)

A. Convolutional layers preserve spatial information between their inputs and outputs.
B. Adding zero-padding around the convolutional layer's input ensures that the pixels at the border receive as much attention as those located elsewhere in the feature map.
C. Convolutional layers in general use fewer parameters than linear layers.

<details>

<summary>Reveal answer</summary>

All of them.

</details>

## Precision & Recall for Multiclass Classification (revisited)

### Computing total value

Sample results for multiclass classification:

```console
              precision    recall  f1-score   support

     class 0       0.50      1.00      0.67         1
     class 1       0.00      0.00      0.00         1
     class 2       1.00      0.67      0.80         3

    accuracy                           0.60         5
   macro avg       0.50      0.56      0.49         5
weighted avg       0.70      0.60      0.61         5
```

`Support` represents the number of instances for each class within the true labels. If the column with `support` has different numbers, then we have class imbalance.

- `macro average` = $\frac{F1_{class1} + F1_{class2} + F1_{class3}}{3}$
- `weighted average` = $\frac{F1_{class1}*SUPPORT_{class1} + F1_{class2}*SUPPORT_{class2} + F1_{class3}*SUPPORT_{class3}}{3}$
- `micro average` = $\frac{F1_{class1}*SUPPORT_{class1} + F1_{class2}*SUPPORT_{class2} + F1_{class3}*SUPPORT_{class3}}{SUPPORT_{class1} + SUPPORT_{class2} + SUPPORT_{class3}}$

To calculate them with `torch`, we can use the classes [`torchmetrics.Recall`](https://lightning.ai/docs/torchmetrics/stable/classification/recall.html) and [`torchmetrics.Precision`](https://lightning.ai/docs/torchmetrics/stable/classification/precision.html):

```python
from torchmetrics import Recall

recall_per_class = Recall(task='multiclass', num_classes=7, average=None)
recall_micro = Recall(task='multiclass', num_classes=7, average='micro')
recall_macro = Recall(task='multiclass', num_classes=7, average='macro')
recall_weighted = Recall(task='multiclass', num_classes=7, average='weighted')
```

When to use each:

- `micro`: imbalanced datasets.
- `macro`: consider errors in small classes as equally important as those in larger classes.
- `weighted`: consider errors in larger classes as most important.

We also have the [F1 score metric](https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html).

<details>

<summary>What does multilabel classification mean?</summary>

When one instance can get multiple classes assigned to it. This is the case, for example, in research article authorship identification: one article has multiple authors.

</details>

### Computing per class value

- We can also analyze the performance per class.
- To do this, compute the metric, setting `average=None`.
  - This gives one score per each class:

```python
print(f1)
```

```console
tensor([0.6364, 1.0000, 0.9091, 0.7917,
        0.5049, 0.9500, 0.5493],
        dtype=torch.float32)
```

- Then, use the `Dataset`'s `.class_to_idx` attribute that maps class names to indices.

```python
dataset_test.class_to_idx
```

```console
{'cirriform clouds': 0,
 'clear sky': 1,
 'cumulonimbus clouds': 2,
 'cumulus clouds': 3,
 'high cumuliform clouds': 4,
 'stratiform clouds': 5,
 'stratocumulus clouds': 6}
```

## Multi-input models

- Models that accept more than one source of data.
- We might want the model to use **multiple information sources**, such as two images of the same car to predict its model.
- **Multi-modal models** can work on different input types such as image and text to answer a question about the image and/or the text.
- In **metric learning**, the model learns whether two inputs represent the same object.
  - Passport control system that compares our passport photo with a picture it takes of us.
- The model can learn that that two augmented versions of the same input represent the same object, thus outputting what the commonalities are or what transformations were applied.

![w04_multi_input.png](assets/w04_multi_input.png "w04_multi_input.png")

## The Omniglot dataset

A collection of images of `964` different handwritten characters from `30` different alphabets.

![w04_omniglot.png](assets/w04_omniglot.png "w04_omniglot.png")

**Task:** Build a two-input model to classify handwritten characters. The first input will be the image of the character, such as this Latin letter `k`. The second input will the the alphabet that it comes from expressed as a one-hot vector.

![w04_omniglot_task.png](assets/w04_omniglot_task.png "w04_omniglot_task.png")

<details>
<summary>How can we solve this?</summary>

Process both inputs separately, then concatenate their representations.

![w04_omniglot_high_level_idea.png](assets/w04_omniglot_high_level_idea.png "w04_omniglot_high_level_idea.png")

The separate processing, would just be us implementing two networks and using them as layers in our one network that will process one sample:

- The image processing network/layer can have the following architecture:
  - several chained convolutional -> max pool -> activation layers;
  - a final linear layer than ouputs a given size, for example `128`.
- The alphabet processing layer can have:
  - several linear -> activation layers;
  - a final linear layer that outputs a given size, for example `8`.
- We can then `fuse` the the outputs in another linear layer by passing the [concatenated output](https://pytorch.org/docs/main/generated/torch.cat.html#torch-cat) from the other two neural networks:

```python
class MyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers1 = nn.ModuleList([
      nn.Linear(32, 64),
      nn.ReLU(),
      nn.Linear(64, 10)
    ])
    self.layers2 = nn.ModuleList([
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1)
    ])
    self.classfier = nn.Linear(10 + 1, 42)
      
  def forward(self, x1, x2):
    for layer in self.layers1:
      x1 = layer(x1)
    for layer in self.layers2:
      x2 = layer(x2)
    x = torch.cat((x1, x2), dim=1)
    return self.classfier(x)
```

</details>

## Multi-output models

- **Predict multiple labels** from the same input, such as a car's make and model from its picture;
- **Multi-label classification**: the input can belong to multiple classes simultaneously:
  - Authorship attribution of a research paper (one paper => many authors).
- In very deep models built of blocks of layers we can add **extra outputs** predicting the same targets after each block.
  - Goal: ensure that the early parts of the model are learning features useful for the task at hand while also serving as a form of regularization to boost the robustness of the network.

![w04_multi_ouput.png](assets/w04_multi_ouput.png "w04_multi_ouput.png")

## Character and alphabet classification

Build a model to predict both the character and the alphabet it comes from based on the image.

![w04_multi_ouput_omniglot.png](assets/w04_multi_ouput_omniglot.png "w04_multi_ouput_omniglot.png")

<details>
<summary>What will be the architecture of the model?</summary>

- Define image-processing sub-network.
- Define output-specific classifiers.
- Pass image through dedicated sub-network.
- Pass the result through each output layer.
- Return both outputs.
- We'll therefore have two loss function objects and two metric tracking objects.

</details>

## Loss weighting

- Now that we have two losses (for alphabets and for characters), we have to choose how to combine them to form the final loss of the model.
- The most intuitive way is to just sum them up:

```python
loss = loss_alpha + loss_char
```

<details>
<summary>What are the advantages of this approach?</summary>

Both classification tasks are deemed equally important.

</details>

<details>
<summary>What are the disadvantages of this approach?</summary>

Both classification tasks are deemed equally important.

</details>

### Varying task importance

Classifing the alphabet should be the easier task, since it has less classes.

<details>
<summary>How could we translate this understanding to the model?</summary>

We could multiply the character loss by a scaler:

```python
loss = loss_alpha + loss_char * 2
```

The above is more intuitive for us, however, it'd be better for the model if the weights sum up to `1`, so let's use the below approach:

```python
loss = 0.33 * loss_alpha + 0.67 * loss_char
```

</details>

### Losses on different scales

Losses must be on the same scale before they are weighted and added.

<details>
<summary>But, why - what problems would we have otherwise?</summary>

Example tasks:

- Predict house price => MSE loss.
- Predict quality: low, medium, high => CrossEntropy loss.

- CrossEntropy loss is typically in the single-digits range.
- MSE loss can reach tens of thousands.
- **Result:** Model would ignore quality assessment task.

</details>

<details>
<summary>How do we solve this?</summary>

Normalize both losses before weighing and adding.

```python
loss_price = loss_price / torch.max(loss_price)
loss_quality = loss_quality / torch.max(loss_quality)
loss = 0.7 * loss_price + 0.3 * loss_quality
```

</details>

## Checkpoint

Three versions of the two-output model for alphabet and character prediction that we discussed have been trained: `model_a`, `model_b`, and `model_c`. For all three, the loss was defined as follows:

```python
loss_alpha = criterion(outputs_alpha, labels_alpha)
loss_char = criterion(outputs_char, labels_char)
loss = ((1 - char_weight) * loss_alpha) + (char_weight * loss_char)
```

Each of the three models was trained with a different `char_weight`: `0.1`, `0.5`, or `0.9`.

Here's what accuracies you have recorded:

```python
evaluate_model(model_a)
```

```console
Alphabet: 0.2808536887168884
Character: 0.1869264841079712
```

```python
evaluate_model(model_b)
```

```console
Alphabet: 0.35044848918914795
Character: 0.01783689111471176
```

```python
evaluate_model(model_c)
```

```console
Alphabet: 0.30363956093788147
Character: 0.23837509751319885
```

Which `char_weight` was used to train which model?

A. `model_a`: `0.1`, `model_b`: `0.5`, `model_c`: `0.9`
B. `model_a`: `0.1`, `model_b`: `0.9`, `model_c`: `0.5`
C. `model_a`: `0.5`, `model_b`: `0.1`, `model_c`: `0.9`
D. `model_a`: `0.9`, `model_b`: `0.1`, `model_c`: `0.5`
C. `model_a`: `0.9`, `model_b`: `0.5`, `model_c`: `0.1`

<details>
<summary>Reveal answer</summary>

C.

Notice how the model with `90%` of its focus on alphabet recognition (`char_weight=0.1`) does very poorly on the character task.

As we increase `char_weight` to `0.5`, the alphabet accuracy drops slightly due to the increased focus on characters, but when it reaches `char_weight=0.9`, the alphabet accuracy increases slightly with the character accuracy, highlighting the synergy between the tasks.

</details>

# Week 05 - Image Processing

**This session is focused on helping you increase the quality of your entire dataset (validation and test included) before you start thinking about modelling it. As such the results from the below techniques have many different downstream applications, only one of which is creating and training deep learning models.**

## Introduction

### Context

We understand now that the quality of images we pass to our models directly affects the quality of the model's predictions. We also want to:

- train a model using a low number of epochs;
- train a model that does not underfit our data;
- train a model that does not overfit our data;
- train the smallest model possible (small and simple architecture) that would do the job (Occam's razor):
  - smaller => faster inferece;
  - smaller => faster training;
  - smaller => faster fine-tuning;
  - smaller => less storage.

### Problem

We start creating and experimenting with multiple models but they all:

- become too big in terms of parameters;
- take long to train and predict;
- don't manage to obtain high metric scores on the validation set.

### Solution

We can try to help the models by **removing unnecessary information in their inputs**. For example, we can:

- leave only the edges of the objects present in the images;
- give the models the contours of the objects instead of the full objects. The model then would:
  - only have to figure out what shape the contours represent
  - not have to deal with extracting the edges and forming contours using them (i.e. we reduce the amount of feature engineering, by doing it beforehand for all incoming data).
- reduce the noise in the images:
  - be it background noise;
  - or just removing objects (parts of the image) that the models should not pay attention to.
- and do a lot more to help get a better representation of our data, by **focusing the model's attention** on what would help it do better.

### Benefits

- We get all the above benefits.
- We get a higher quality dataset.
- Because the preprocessing it predefined we know what task the model will be solving (because we do the feature engineering).

### Context

We want to transform our images:

- be it their entire representation (ex. going from a matrix to a vector of points);
- or by altering how (and how much) information they represent.

### Problem

We don't know a library that can do that.

### Solution - scikit-image

![w05_skimage_logo.png](assets/w05_skimage_logo.png "w05_skimage_logo.png")

- Easy to use.
- Makes use of Machine Learning.
- Out of the box / Built-in complex algorithms.

### Images in scikit-image

The library has a lot of built-in images. Check them out [in the data module](https://scikit-image.org/docs/stable/api/skimage.data.html#).

```python
from skimage import data
rocket_image = data.rocket()
print(type(rocket_image))
print(rocket_image.shape)
print(rocket_image[0])
```

```console
<class 'numpy.ndarray'>
(427, 640, 3)
[[17 33 58]
 [17 33 58]
 [17 33 59]
 ...
 [ 8 19 37]
 [ 8 19 37]
 [ 7 18 36]]
```

Display images:

```python
def show_image(image, title='Image', cmap_type='gray'):
  plt.imshow(image, cmap=cmap_type)
  plt.title(title)
  plt.axis('off')
  plt.show()

show_image(rocket_image)
```

![w05_rocket_img.png](assets/w05_rocket_img.png "w05_rocket_img.png")

## RGB and Grayscale

### Context

Often we want to reduce the information present by converting to grayscale.

It can also be the case that we want to go back from grayscale to RGB.

Grayscale:

![w05_grayscale.png](assets/w05_grayscale.png "w05_grayscale.png")

RGB:

![w05_rgb_breakdown.png](assets/w05_rgb_breakdown.png "w05_rgb_breakdown.png")

### Problem

How would we do that?

### Solution - `color.rgb2gray` and `color.gray2rgb`

Convert between the two:

```python
from skimage import data, color
original = data.astronaut()
grayscale = color.rgb2gray(original)
rgb = color.gray2rgb(grayscale)
```

```python
show_image(original)
```

![w05_img_orig.png](assets/w05_img_orig.png "w05_img_orig.png")

```python
show_image(grayscale)
```

![w05_img_gray.png](assets/w05_img_gray.png "w05_img_gray.png")

The value of each grayscale pixel is calculated as the weighted sum of the corresponding red, green and blue pixels as:

```text
Y = 0.2125 R + 0.7154 G + 0.0721 B
```

```python
show_image(rgb)
print(f'{grayscale.shape=}')
print(f'{rgb.shape=}')
```

![w05_img_rgb.png](assets/w05_img_rgb.png "w05_img_rgb.png")

```console
grayscale.shape=(512, 512)
rgb.shape=(512, 512, 3)
```

Note that `color.gray2rgb` just duplicates the gray values over the three color channels (as it cannot know the true intensities).

## Basic image operations

### Using `numpy`

Because the type of the loaded images is `<class 'numpy.ndarray'>`, we can directly apply `numpy` manipulations.

#### Vertical flip

```python
show_image(np.flipud(original))
```

![w05_flipud.png](assets/w05_flipud.png "w05_flipud.png")

#### Horizontal flip

```python
show_image(np.fliplr(original))
```

![w05_fliplr.png](assets/w05_fliplr.png "w05_fliplr.png")

### The `transform` module

For more complex transformations, we can use the `skimage` library instead:

#### Rotating

```python
from skimage import transform
transform.rotate(image, -90)
```

![w05_rotate.png](assets/w05_rotate.png "w05_rotate.png")

#### Rescaling

```python
from skimage import transform
transform.rescale(image, 1/4, anti_aliasing=True)
```

![w05_scaled.png](assets/w05_scaled.png "w05_scaled.png")

#### Resizing

```python
from skimage import transform
height = 400
width = 500
transform.resize(image, (height, width), anti_aliasing=True)
```

![w05_resized.png](assets/w05_resized.png "w05_resized.png")

##### Problem

When we rescale/reside by a large factor:

- we lose the bondaries of the objects;
- background and foreground objects merge (the distinction is lost).

![w05_alasing.png](assets/w05_alasing.png "w05_alasing.png")

![w05_aliasing_comparison.png](assets/w05_aliasing_comparison.png "w05_aliasing_comparison.png")

##### Solution - anti-aliasing

The is because of the [Aliasing phenomenon](https://en.wikipedia.org/wiki/Aliasing): a reconstructed signal from samples of the original signal contains low frequency components that **are not present in the original one**.

- Aliasing makes the image look like it has waves or ripples radiating from a certain portion.
  - This happens because the pixelation of the image is poor.

We can then set the `anti_aliasing` parameter to `True`.

## Thresholding

### Context

- We have a task for detecting whether there is a human in a picture.
- We decide that it would be best to reduce the information in an image so as to get a separation between foreground and background.
  - We want all background parts to have `0` and all foreground to have `1`.

![w05_inverted_thresholding_example.png](assets/w05_inverted_thresholding_example.png "w05_inverted_thresholding_example.png")

In essence, this is a type of image segmentation for object detection. However, we may also have different applications:

- Object detection;
- Face detection;
- Noise removal;
- etc.

![w05_thresholding_example.png](assets/w05_thresholding_example.png "w05_thresholding_example.png")

### Problem

How do we do that?

### Simple solution - global / histogram based thresholding

- **Image histogram:** A graphical representation of the amount of pixels of each intensity value.
  - From `0` (pure black) to `255` (pure white).

![w05_histogram.png](assets/w05_histogram.png "w05_histogram.png")
![w05_histogram_color.png](assets/w05_histogram_color.png "w05_histogram_color.png")

- Goal: Partition an image into a foreground and background, by setting one threshold for all pixels.
- Best for **bimodal** histograms.
- Steps:
  1. Convert to grayscale.
  2. Obtain a threshold value: either manually by looking at the histogram or by using an algorithm.
  3. Set each pixel to:
     - `255` (white) if `value > thresh`
     - `0`, otherwise

We can also automatically try out several global thresholding algorithms. This can be useful when we don't want to manually experiment and choose a value.

```python
from skimage.filters import try_all_threshold
fig, axis = try_all_threshold(grayscale, verbose=False)
```

![w05_global_algorithms.png](assets/w05_global_algorithms.png "w05_global_algorithms.png")

Once we like what one of the algorithms has produced we can instantiate it manually:

```python
from skimage.filters import threshold_otsu
thresh = threshold_otsu(image)
binary_global = image > thresh
show_image(image, 'Original')
show_image(binary_global, 'Global thresholding')
```

![w05_global_otsu.png](assets/w05_global_otsu.png "w05_global_otsu.png")

### Advanced solutions: local / adaptive thresholding

- Determine based on surronding values up to a given range.
- Good for uneven background illumination.
- The threshold value is the weighted mean for the local neighborhood of a pixel subtracted by a constant [reference](https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_local).

![w05_global_local_thresholding.png](assets/w05_global_local_thresholding.png "w05_global_local_thresholding.png")

```python
from skimage.filters import threshold_local
thresh = threshold_local(text_image, block_size=35, offset=10)
binary_global = text_image > thresh
show_image(text_image, 'Original')
show_image(binary_global, 'Local thresholding')
```

![w05_local.png](assets/w05_local.png "w05_local.png")

## Edge detection

### Context

We have a task about creating a deep learning model that can count the number of coins in an image.

We don't need to know the value of the coints - just their count.

### Problem

How can we increase the quality of our data to help the model?

### Solution - Sobel filter

We can use the [Sobel filter](https://en.wikipedia.org/wiki/Sobel_operator):

![w05_filter_sobel.png](assets/w05_filter_sobel.png "w05_filter_sobel.png")

![w05_filter_sobel2.png](assets/w05_filter_sobel2.png "w05_filter_sobel2.png")

The Sobel filter is [built-in](https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel) in scikit-image. Whenever, you're using it, make sure the image is grayscaled first.

Just like in convolutional neural networks, filters work on a neighborhood of points.

![w05_filter_neighborhood.png](assets/w05_filter_neighborhood.png "w05_filter_neighborhood.png")

For example the Sobel filter uses the following two filters:

![w05_filter_sobel_matrices.png](assets/w05_filter_sobel_matrices.png "w05_filter_sobel_matrices.png")

### Advanced solution - edge detection with the [Canny algorithm](https://en.wikipedia.org/wiki/Canny_edge_detector)

Let's say that we do want to also get the sum of the coints. Then Sobel might not work for us. We would have to use a more complex algorithm.

The process of Canny edge detection algorithm can be broken down to four different steps:

1. Smooth the image using a Gaussian with `sigma` width.
2. Apply the horizontal and vertical Sobel operators to get the gradients within the image. The edge strength is the norm of the gradient.
3. Thin potential edges to `1`-pixel wide curves. First, find the normal to the edge at each point. This is done by looking at the signs and the relative magnitude of the `X`-Sobel and `Y`-Sobel to sort the points into `4` categories: `horizontal`, `vertical`, `diagonal` and `antidiagonal`. Then look in the normal and reverse directions to see if the values in either of those directions are greater than the point in question. Use interpolation to get a mix of points instead of picking the one that’s the closest to the normal.
4. Perform a hysteresis thresholding: first label all points above the high threshold as edges. Then recursively label any point above the low threshold that is `8`-connected to a labeled point as an edge.

Here is a visual walkthorugh:

![w05_canny_walkthrough.png](assets/w05_canny_walkthrough.png "w05_canny_walkthrough.png")

Compared to the Sobel algorithm, Canny:

- can detect more complex edges;
- is faster.

Thus, Canny is widely considered to be the standard edge detection method in image processing:

![w05_canny_vs_sobel.png](assets/w05_canny_vs_sobel.png "w05_canny_vs_sobel.png")

We can use the function [canny](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.canny) to perform edge detection using the Canny algorithm.

Two parameters should be considered when using it:

- `image`: This is the input image. Note that it should be a **2D array**, i.e. **is has to be converted to grayscale first**.
- `sigma`: This is the standard deviation of the Gaussian filter that is applied on step `1` of the execution of the algorithm.
  - It can be any positive floating point value.
  - The higher it is, the less edges are going to be detected, since more aggressive smoothing will be applied. The default value of `1` often works pretty well.

## Contrast enhancement

### Context

We have a task about detecting lung disease:

![w05_contrast_medical.png](assets/w05_contrast_medical.png "w05_contrast_medical.png")

### Problem

The images that we receive do not show the details well. They are not very contrastive.

The contrast is the difference between the maximum and minimum pixel intensity in the image.

![w05_enistein.png](assets/w05_enistein.png "w05_enistein.png")

- An image of low contrast has small difference between its dark and light pixel values.
  - Is usually skewed either to the right (being mostly light), to the left (when is mostly dark), or located around the middle (mostly gray).

![w05_contrast_light.png](assets/w05_contrast_light.png "w05_contrast_light.png")

### Solutions

- **Contrast stretching**: stretches the histogram so the full range of intensity values of the image is filled.
- **Histogram equalization (HE)**: spreads out the most frequent histogram intensity values using a probability distribution.
  - Standard HE;
  - Adaptive HE (also known as AHE);
  - Contrast Limited Adaptive HE (also known as [CLAHE](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization)).

![w05_he.png](assets/w05_he.png "w05_he.png")

#### Standard Histogram Equalization

```python
from skimage import exposure
image_eq = exposure.equalize_hist(image)
```

Sometimes this works well:

![w05_she_positive.png](assets/w05_she_positive.png "w05_she_positive.png")

But other times, we get a result that, despite the increased contrast, doesn't look natural. In fact, it doesn't even look like the image has been enhanced at all.

![w05_she.png](assets/w05_she.png "w05_she.png")

#### Contrastive Limited Adaptive Equalization

We can then utilize the CLAHE method to obtain a better representation of our image:

![w05_clahe.png](assets/w05_clahe.png "w05_clahe.png")

```python
from skimage import exposure
exposure.equalize_adapthist(image, clip_limit=0.03)
```

![w05_clahe2.png](assets/w05_clahe2.png "w05_clahe2.png")

### Checkpoint

What is the contrast of the following image?

![w05_checkpoint_image.png](assets/w05_checkpoint_image.png "w05_checkpoint_image.png")

![w05_checkpoint_hist.png](assets/w05_checkpoint_hist.png "w05_checkpoint_hist.png")

A. The contrast is `255` (high contrast).
B. The contrast is `148`.
C. The contrast is `189`.
D. The contrast is `49` (low contrast).

<details>
<summary>Reveal answer</summary>

B.

It can be inferred from the histogram:

- The maximum value is almost `250` (`247` in reality).
- The minimum value is around `100` (`99` in reality).

$247 - 99 = 148$

</details>

## Image Morphology

<details>
<summary>What is image morphology?</summary>

The study of the shapes/textures of objects in an image.

</details>

<details>
<summary>What task can image morphology help with?</summary>

- Object detection.
- Optical character recognition.

</details>

<details>
<summary>Does image morphology work best with binary, grayscale or color images?</summary>

- It is typically applied after (binary) thresholding, so is works best with binary images.
- Can be extended to [grayscale](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=8ae9fc1e08c790f737d52c4ab6e20234aa269faa).

</details>

<details>
<summary>What problems might binary images have that could be addressed by image morphology?</summary>

They can often be distorted by noise and texture:

![w05_binary_distorted.png](assets/w05_binary_distorted.png "w05_binary_distorted.png")

Simple tasks can start to become complex:

![w05_r5.png](DATA/w05_r5.png "w05_r5.png")

</details>

<details>
<summary>What are the two most basic morphological operations?</summary>

Dilation:

![w05_dilation.png](assets/w05_dilation.png "w05_dilation.png")

and erosion:

![w05_erosion.png](assets/w05_erosion.png "w05_erosion.png")

<details>
<summary>Looking at the examples what does dilation do?</summary>

It adds pixels to the boundaries of objects in an image.

</details>

<details>
<summary>Looking at the examples what does erosion do?</summary>

It removes pixels on object boundaries.

</details>

</details>

<details>
<summary>What is the "structuring element"?</summary>

- It defines the number of pixels added or removed.
- It's implemented as a small binary image used to `probe` the input image.
- The input image should be padded:
  - with `1`s, if the operation is erosion.
  - with `0`s, if the operation is dilation.

We try to `fit` it in the object we want to apply erosion or dilation to.

![w05_structuring_element.png](assets/w05_structuring_element.png "w05_structuring_element.png")

Dilation then is the set of all points where the structuring element `touches` or `hits` the foreground. If that is the case, `1` is outputted at the origin of the structuring element.

Erosion is the set of all points in the image where the structuring element fits **entirely**. If that is the case, `1` is outputted at the origin of the structuring element.

</details>

## Checkpoint

Compute dilation on the image using the structuring element.

![w05_dilation_checkpoint.png](assets/w05_dilation_checkpoint.png "w05_dilation_checkpoint.png")

<details>
<summary>Reveal answer</summary>

![w05_dilation_answer.png](assets/w05_dilation_answer.png "w05_dilation_answer.png")

You can play around with dilation and erosion [here](https://animation.geekosophers.com/morphological-operations/Dilation/Dilation%20Animation%20Js/index.html) and [here](https://animation.geekosophers.com/morphological-operations/Erosion/Erosion%20Animation%20Js/index.html).

</details>

## Shapes in `scikit-image`

To create structuring elements we can use the function [`footprint_rectangle`](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.footprint_rectangle).

```python
from skimage import morphology
morphology.footprint_rectangle((4, 4))
```

```console
array([[1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1]], dtype=uint8)
```

We can then use the functions [`binary_erosion`](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.binary_erosion) and [`binary_dilation`](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.binary_dilation).

## Restoring images with [inpainting](https://en.wikipedia.org/wiki/Inpainting)

We can restore damaged sections of an image, provided it has regions of the same context as the missing piece(s).

> **Definition:** **Inpainting** is a conservation process where damaged, deteriorated, or missing parts of an artwork are filled in to present a complete image.

![w05_image_restore.png](assets/w05_image_restore.png "w05_image_restore.png")

To do this in `scikit-image` we can use the function [inpaint_biharmonic](https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.inpaint_biharmonic). It takes two arguments:

- `image`: the image to be inpainted;
- `mask`: a NumPy array of `0`s and `1`s. The regions with `1`s should correspond to the regions in `image` that should be inpainted. Thus, the sizes of `mask` and `image` should match.
- `channel_axis` is an optional, but also an important parameter. If `None`, the image is assumed to be a **grayscale** (single channel) image. Otherwise, this parameter indicates **which axis of the array corresponds to channels**.

## Denoising images

- Noise is the result of errors in the image acqusition process.
- Results in pixel values that do not reflect the true intensities of the real scene.

![w05_noise_dog.png](assets/w05_noise_dog.png "w05_noise_dog.png")

In scikit-learn, we can introduce noise in an image using the function [`random_noise`](https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.random_noise). It only requires the image to be passed in.

### Simple solution - Gaussian smoothing

Examples for Gaussian smoothing (also [built-in](https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.gaussian)):

![w05_gauss1.png](assets/w05_gauss1.png "w05_gauss1.png")

![w05_gauss2.png](assets/w05_gauss2.png "w05_gauss2.png")

![w05_gauss3.png](assets/w05_gauss3.png "w05_gauss3.png")

### Advanced solutions

There are several strategies to remove noise if it's harder to pinpoint it:

- Total variation (TV): minimize the total variation of the image. It tends to produce cartoon-like images, that is, piecewise-constant images, thus any edges in the image are lost;
- Bilateral: replaces the intensity of each pixel with a weighted average of intensity values from nearby pixels. Use when the goal is to preserve the edges in the image;
- [Wavelet](https://scikit-image.org/docs/stable/auto_examples/filters/plot_denoise_wavelet.html);
- [Non-local means](https://en.wikipedia.org/wiki/Non-local_means).

![w05_tv_vs_bilateral.png](assets/w05_tv_vs_bilateral.png "w05_tv_vs_bilateral.png")

We can apply TV denoising using the function [`denoise_tv_chambolle`](https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_tv_chambolle) and bilateral filtering using the function [`denoise_bilateral`](https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_bilateral). Note that both of them have the argument `channel_axis`.

## Segmentation

<details>
<summary>What is segmentation?</summary>

The process of partitioning images into regions (called segments) to simplify and/or change the representation into something more meaniningful and easier to analyze.

![w05_pure_segmenation.png](assets/w05_pure_segmenation.png "w05_pure_segmenation.png")

</details>

<details>
<summary>What is semantic segmentation?</summary>

Classifying each pixel as belonging to a certain class from a prefined set of classes. This task is usually solved with deep neural networks of type [Autoencoder](https://en.wikipedia.org/wiki/Autoencoder).

![w05_semantic_segmenation.jpeg](assets/w05_semantic_segmenation.jpeg "w05_semantic_segmenation.jpeg")

</details>

There are two types of segmentation:

- Supervised: we specify the threshold value ourselves;
- Unsupervised: algorithms that subdivide images into meaningful regions automatically.
  - The user may still be able to tweak certain settings to obtain the desired output.

<details>
<summary>What is an example of an unsupervised algorithm for segmentation we saw earlier?</summary>

The otsu thresholding algorithm.

</details>

### Superpixels

- We'll look into segmentation first without neural networks.
- Since looking at single pixels by themselves is a hard task that is solved using neural networks, we'll use groups of pixels.

> **Definition:** A region of pixels is called a **superpixel**.

![w05_superpixel.png](assets/w05_superpixel.png "w05_superpixel.png")

### Simple Linear Iterative Clustering ([SLIC](https://mrl.cs.vsb.cz/people/gaura/ano/slic.pdf))

- Segments the image using K-Means clustering.
- Takes in all the pixel values of the image and tries to separate them into a predefined number of sub-regions.
- In `scikit-image` it is implemented in the function [`slic`](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.slic).
  - Also has a parameter `channel_axis`.
  - Returns an array that is the same shape as the original image in which each pixel is assigned a label.

```python
import matplotlib.pyplot as plt
from skimage import segmentation, color, data

segments = segmentation.slic(data.astronaut(), n_segments=300, channel_axis=-1)
segments
```

```console
array([[ 1,  1,  1, ...,  9,  9,  9],
       [ 1,  1,  1, ...,  9,  9,  9],
       [ 1,  1,  1, ...,  9,  9,  9],
       ...,
       [51, 51, 51, ..., 56, 56, 56],
       [51, 51, 51, ..., 56, 56, 56],
       [51, 51, 51, ..., 56, 56, 56]], shape=(512, 512))
```

We can then use the function [label2rgb](https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.label2rgb). It'll return an image where the segments obtained are highlighted, either with random colors or with the average color of the superpixel segment.

```python
segmented_image = color.label2rgb(segments, data.astronaut(), kind='avg')
plt.axis('off')
plt.imshow(segmented_image)
plt.show()
```

![w05_astro_after_kmeans.png](assets/w05_astro_after_kmeans.png "w05_astro_after_kmeans.png")

## Image contours

> **Definition:** A contour is a closed shape of points or line segments, representing the boundaries of an object. Having multiple contours means there are multiple objects.

Using image contours will help in:

- measuring object size;
- classifying shapes;
- determining the number of objects in an image.

![w05_contours.png](assets/w05_contours.png "w05_contours.png")

The input to a contour-finding function should be a **binary image**, which we can produce by first applying thresholding or edge detection. The objects we wish to detect should be white, while the background - black.

![w05_thresholded_before_contour.png](assets/w05_thresholded_before_contour.png "w05_thresholded_before_contour.png")

We can then use the function [find_contours](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.find_contours) passing in the binary image. The function:

- Joins pixels of equal brightness in a 2D array above a given `level` value (default = `(max(image) + min(image)) / 2`).
  - the closer `level` is to `1`, the more sensitive the method is to detecting contours, so more complex contours will be detected.
  - We have to find the value that best detects the contours we care for.
- Returns a list of arrays where each array holds the contours of an object as pairs of coordinates.

![w05_constant_level.png](assets/w05_constant_level.png "w05_constant_level.png")

The shapes of the contours can tell us which object they belong to. We can then use them to count the objects from a particular shape.

![w05_contour_shape.png](assets/w05_contour_shape.png "w05_contour_shape.png")

## Corner detection

### Corners

Detecting corners can pop up as a subtask when doing:

- motion detection;
- video tracking;
- panorama stitching;
- 3D modelling;
- object detection;
- image registration;
- etc, etc.

![w05_registration.png](assets/w05_registration.png "w05_registration.png")

> **Definition:** A corner is the intersection of two edges.

Intuitively, it can also be a junction of contours. We can see some obvious corners in this checkerboard image and in this building image on the right.

![w05_corners.png](assets/w05_corners.png "w05_corners.png")

> **Definition:** Points of interest are groups of pixels in an image which are invariant to rotation, translation, intensity, and scale changes.

By detecting corners (and edges) as interest points, we can match objects from different perspectives.

![w05_corner_matching.png](assets/w05_corner_matching.png "w05_corner_matching.png")

![w05_corner_matching2.png](assets/w05_corner_matching2.png "w05_corner_matching2.png")

### [Harris corner detector](https://en.wikipedia.org/wiki/Harris_corner_detector)

Commonly, the Harris corner detector algorithm can be divided into five steps:

1. Color to grayscale.
2. Spatial derivative calculation.
3. Structure tensor setup.
4. Harris response calculation.
5. Non-maximum suppression.

The first `4` steps are implemented in `scikit-image` as the function [corner_harris](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.corner_harris).

It returns the Harris measure response image, i.e., the resulting image shows only the approximated points where the corner-candidates are.

![w05_harris_return_value.png](assets/w05_harris_return_value.png "w05_harris_return_value.png")

To find the corners in the measure response image, we need to perform non-maximum suppression. For that we can use the function [corner_peaks](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.corner_peaks). It:

- Has an optional parameter `min_distance` that sets the minimum distance between peaks. Default value: `1`.
- Has another optional parameter `threshold_rel` sets the minimum intensity of peaks. Default value: `0` (consider all candidate-peaks).
- Returns a list of 2D arrays with corner coordinates.
- Can be applied directly on the return value of the Harris algorithm:

```python
coords = corner_peaks(corner_harris(image), min_distance=5)
print(f'A total of {len(coords)} corners were detected.')
print(coords[:5])
```

```console
A total of 1267 corners were detected.
[[445 310]
 [368 400]
 [455 346]
 [429 371]
 [467 241]]
```

We can then plot those points via `matplotlib` on top of the original image.

```python
plt.plot(coords[:, 0], coords[:, 1], '+r', markersize=15)
```

## Face detection

We can detect faces using the class [Cascade](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.Cascade).

```python
detector = Cascade(xml_file=trained_file)
```

The parameter `xml_file` must be set to a file (or path to a file) that holds parameters of a trained model.

Since we're not focusing on doing this right now, we'll use an already trained and built-in model in `scikit-image` using the function [lbp_frontal_face_cascade_filename](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.lbp_frontal_face_cascade_filename).

```python
trained_file = data.lbp_frontal_face_cascade_filename()
detector = Cascade(xml_file=trained_file)
```

We can then use the method [detect_multi_scale](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.Cascade.detect_multi_scale) to perform face detection.

- It creates a window that will be moving through the image until it finds something similar to a human face:
  - the parameter `img` can be a grayscale or color image;
  - the parameter `scale_factor` sets by how much the search window will expand after one convolution;
  - the parameter `step_ratio` sets by how much the search window will shift to the right and bottom. `1` means exhaustive search. Values in the interval `[1 .. 1.5]` give good results.
  - the parameter `min_size` sets the minimum size of the window that will be searching for the farthest faces.
  - the parameter `max_size` sets the maximum size of the window that will be searching for the closest faces.
- Searching happens on multiple scales. The window will have a minimum size to spot far-away faces and a maximum size to also find the closer faces in the image.
- The return value is a list of bounding boxes defined by the:
  - upper left coordinates: `r` and `c`;
  - the width and height of the bounding box: `width` and `height`.

![w05_multiscale_fd.png](assets/w05_multiscale_fd.png "w05_multiscale_fd.png")

```python
detected = detector.detect_multi_scale(img=image, scale_factor=1.2, step_ratio=1, min_size=(10, 10), max_size=(200, 200))
print(detected)
```

```console
[{'r': 115, 'c': 210, 'width': 167, 'height': 167}]
```

## Applications

There are many situations in which we can apply what we've discussed here:

- Converting an image to grayscale;
- Detecting edges;
- Detecting corners;
- Reducing noise;
- Restoring images;
- Approximating objects' sizes;
- Privary protection;
- etc, etc.

Let's look at how we would solve a privacy protection case. We want to turn this:

![w05_privacy_start.png](assets/w05_privacy_start.png "w05_privacy_start.png")

into this:

![w05_privacy_end.png](assets/w05_privacy_end.png "w05_privacy_end.png")

<details>
<summary>What steps should we take?</summary>

1. Detect faces.
2. Cut out faces.
3. Apply a Gaussian filter on each face with large `sigma` to introduce blurriness.
4. Stitch the faces back to the original image.

</details>
