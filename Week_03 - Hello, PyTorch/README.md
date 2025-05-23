# 🎯 Goals for week 03

1. Get familiar with the deep learning framework `PyTorch`.
2. Create your first neural network with `PyTorch`.
3. Play around with different architectures and modify hyperparameters.
4. See how you can evaluate and improve a model.
5. Practice writing high quality code:
   1. Easy to read.
   2. Safe from bugs.
   3. Ready for change.

## Task 1

**Description:**

Creating a PyTorch `Tensor` object from a Python list of temperature data from two weather stations.

```python
[[72, 75, 78], [70, 73, 76]]
```

The list has two sublists whose elements represent a different day each, with columns for readings from two stations.

Output:

- the created `Tensor` object;
- its shape;
- its data type.

While continuing your temperature data collection, you realize that the recorded temperatures are off by `2` degrees, so you need to add `2` degrees to the tensor of temperatures. Output the corrected version.

**Acceptance criteria:**

1. Output is the same as in the test case.

**Test case:**

```python
python task01.py
```

```console
Temperatures: tensor([[72, 75, 78],
        [70, 73, 76]])
Shape of temperatures: torch.Size([2, 3])    
Data type of temperatures: torch.int64       
Corrected temperatures: tensor([[74, 77, 80],
        [72, 75, 78]])
```

## Task 2

**Description:**

Create a neural network using the `Sequential` class that processes the below input using two `Linear` layers. Use any output dimension for the first layer you want. Output the resulting `logit`.

```python
temperature_observation = [2, 3, 6, 7, 9, 3, 2, 1]
```

**Acceptance criteria:**

1. The `Sequential` class is used.
2. Two `Linear` layers are present.
3. Batch size is `1`.
4. The `logit` is outputted.

**Test case:**

Output value may be different.

```python
python task02.py
```

```console
tensor([[0.9504]], grad_fn=<AddmmBackward0>)
```

## Task 3

**Description:**

Create a neural network with one linear layer that takes a temperature observation and returns an output that represents how confident the network is that the current season is spring.

Only define the network and apply it directly (i.e. without training) on the below observation.

```python
temperature_observation = [3, 4, 6, 2, 3, 6, 8, 9]
```

Write the letter of the answer to the following question in a comment:

*Which of the following is **false** about the output returned by your binary classifier?*

A. We can use a threshold of 0.5 to determine if the output belongs to one class or the other.
B. It can return any float value.
C. It is produced from an untrained model so it is not yet meaningful.

**Acceptance criteria:**

1. The architecture in the description is created.
2. Appropriate layers are used.
3. A comment is present with the letter of the correct option to the question in the description.

**Test case:**

Output value may be different.

```python
python task03.py
```

```console
0.05850396305322647
```

## Task 4

**Description:**

Calculate the cross entropy for the following output of a neural network, assuming the correct index label is `2`.

```python
y = [2]
scores = torch.tensor([[0.1, 6.0, -2.0, 3.2]])
```

**Acceptance criteria:**

1. The test case passes.
2. The variables are used as they are given in the description and not changed.

**Test case:**

```python
python task04.py
```

```console
tensor(8.0619, dtype=torch.float64)
```

## Task 5

**Description:**

Loading your data into a PyTorch `Dataset` object will be one of the first steps you take in order to create and train a neural network with PyTorch.

The `TensorDataset` class is helpful when your dataset can be loaded directly as a NumPy array. Create such a dataset from a NumPy array with 12 rows and 9 columns with randomly chosen values from a uniform distribution over $[0, 1)$. Treat the last column as the target variable. Output the last sample and label in the created `Dataset` object.

**Acceptance criteria:**

1. A NumPy array of shape `(12, 9)` is created.
2. The NumPy array contains random floating-point values from a uniform distribution over $[0, 1)$.

**Test case:**

Due to randomness while drawing values, your output may be different.

```python
python task05.py
```

```console
Last sample: tensor([0.0471, 0.1919, 0.6886, 0.4382, 0.5270, 0.8424, 0.2288, 0.5889],
       dtype=torch.float64)
Last label: tensor([0.4969], dtype=torch.float64)
```

## Task 6

**Description:**

Load the `ds_salaries.csv` (present in our `DATA` folder in the root of this repository) dataset using [`pandas`](https://pandas.pydata.org/). It contains salaries of data scientists. We'll use a subset of the features to predict salaries of new observations - `salary_in_usd`. Use the features: `experience_level`, `employment_type`, `remote_ratio` and `company_size` to create a `Dataset` and `DataLoader` objects. Normalize the features and the target and encode the categorical variables using `sklearn`.

Create and train a neural network with however many layers you decide (it does not have to be deep - even two layers will be perfectly fine). Use the full data for training.

Training process:

- Train for `20` epochs.
- Use a batch size of `8`.
- Use a learning rate of `0.001`.
- Use the `AdamW` optimizer.

Apart from practicing writing the training loop, we'll also this task to do a comparison between different activation functions that are used ***between*** layers.

Make three copies of your network:

- In the first copy, use the `Sigmoid()` function between each of your layers.
- In the second copy, use the `ReLU()` function between each of your layers.
- In the third copy, use the `LeakyReLU()` function between each of your layers.

Plot the average training mean square error loss per epoch for each of them and output the model that obtained the lowest loss on the final epoch. Use `tqdm` to report progress through batches.

**Acceptance criteria:**

1. Only the specified features are used.
2. Normalization is applied to the features and the target.
3. Encoding is applied.
4. All print statements shown in the test case below are present.

**Test case:**

Due to differences in architecture, your output may be different.

```python
python task06.py
```

```console



Training model: nn_with_sigmoid
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 541.60it/s]
Epoch [1/20]: Average loss: 9.353534598586564e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 547.19it/s]
Epoch [2/20]: Average loss: 7.078851902215373e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 542.15it/s]
Epoch [3/20]: Average loss: 7.056489354812689e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 543.55it/s]
Epoch [4/20]: Average loss: 7.053150508545423e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 556.21it/s]
Epoch [5/20]: Average loss: 7.060699831947674e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 554.21it/s]
Epoch [6/20]: Average loss: 6.887961599086842e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 554.80it/s]
Epoch [7/20]: Average loss: 6.92375330138345e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 546.41it/s]
Epoch [8/20]: Average loss: 6.774466400518567e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 549.15it/s]
Epoch [9/20]: Average loss: 6.733088347406078e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 540.98it/s]
Epoch [10/20]: Average loss: 6.571618571818156e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 546.67it/s]
Epoch [11/20]: Average loss: 6.502323731274509e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 539.61it/s]
Epoch [12/20]: Average loss: 6.541735572563099e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 540.96it/s]
Epoch [13/20]: Average loss: 6.256225829646984e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 541.60it/s]
Epoch [14/20]: Average loss: 6.212907061476523e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 545.87it/s]
Epoch [15/20]: Average loss: 6.378607664306278e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 536.47it/s]
Epoch [16/20]: Average loss: 6.150501401928951e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 545.33it/s]
Epoch [17/20]: Average loss: 6.0617692294468956e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 543.70it/s]
Epoch [18/20]: Average loss: 5.966893914278855e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 542.30it/s]
Epoch [19/20]: Average loss: 5.898609333268732e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 552.70it/s]
Epoch [20/20]: Average loss: 5.8306023176566034e-05



Training model: nn_with_relu
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 551.49it/s]
Epoch [1/20]: Average loss: 0.008204216834468404
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 543.87it/s]
Epoch [2/20]: Average loss: 0.0001740022297965086
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 520.41it/s]
Epoch [3/20]: Average loss: 0.00011477003228061763
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 531.24it/s]
Epoch [4/20]: Average loss: 9.021632804427419e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 539.06it/s]
Epoch [5/20]: Average loss: 7.86146015874489e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 543.95it/s]
Epoch [6/20]: Average loss: 7.216201959024263e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 536.61it/s]
Epoch [7/20]: Average loss: 6.791415351770223e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 540.42it/s]
Epoch [8/20]: Average loss: 6.457202337510668e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 535.46it/s]
Epoch [9/20]: Average loss: 6.229478607785515e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 540.96it/s]
Epoch [10/20]: Average loss: 6.019338090576992e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 528.47it/s]
Epoch [11/20]: Average loss: 5.870186558244001e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 529.20it/s]
Epoch [12/20]: Average loss: 5.720063203522207e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 543.30it/s]
Epoch [13/20]: Average loss: 5.5900338669105506e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 535.95it/s]
Epoch [14/20]: Average loss: 5.534962809579906e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 538.25it/s]
Epoch [15/20]: Average loss: 5.3458923948069436e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 538.68it/s]
Epoch [16/20]: Average loss: 5.2153479846927614e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 540.18it/s]
Epoch [17/20]: Average loss: 4.9774081982585564e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 536.75it/s]
Epoch [18/20]: Average loss: 5.0382628060069785e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 536.18it/s]
Epoch [19/20]: Average loss: 4.754270490966132e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 534.78it/s]
Epoch [20/20]: Average loss: 4.425195528223499e-05



Training model: nn_with_leakyrelu
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 555.34it/s]
Epoch [1/20]: Average loss: 0.0067619663006253375
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 553.93it/s]
Epoch [2/20]: Average loss: 0.00010102234957636763
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 543.44it/s]
Epoch [3/20]: Average loss: 8.42549138392895e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 540.80it/s]
Epoch [4/20]: Average loss: 7.647090967094648e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 544.06it/s]
Epoch [5/20]: Average loss: 7.171556330514187e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 539.21it/s]
Epoch [6/20]: Average loss: 6.866294021664728e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 530.06it/s]
Epoch [7/20]: Average loss: 6.634586525841107e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 543.79it/s]
Epoch [8/20]: Average loss: 5.745742021057343e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 544.27it/s]
Epoch [9/20]: Average loss: 5.1883643399779876e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 537.07it/s]
Epoch [10/20]: Average loss: 4.738284425836707e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 533.81it/s]
Epoch [11/20]: Average loss: 4.220122813371273e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 540.27it/s]
Epoch [12/20]: Average loss: 3.883428442865191e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 535.58it/s]
Epoch [13/20]: Average loss: 3.2724280264372515e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 540.95it/s]
Epoch [14/20]: Average loss: 2.965086544028986e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 542.89it/s]
Epoch [15/20]: Average loss: 2.5310745548549983e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 547.51it/s]
Epoch [16/20]: Average loss: 2.086114368462605e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 549.06it/s]
Epoch [17/20]: Average loss: 1.8522638550728383e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 531.72it/s]
Epoch [18/20]: Average loss: 1.3637417812972786e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 547.50it/s]
Epoch [19/20]: Average loss: 1.2083197813770414e-05
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 470/470 [00:00<00:00, 545.94it/s]
Epoch [20/20]: Average loss: 9.449498383739387e-06
Lowest loss of 9.449498383739387e-06 was achieved by model nn_with_leakyrelu.
```

and the following plot is produced:

![w03_task6.png](../assets/w03_task6.png "w03_task6.png")

## Task 7

**Description:**

We'll use this task to practice creating networks with a given capacity threshold in mind.

Create a `3`-layer linear neural network with strictly less than `120` parameters, using `8` features as input and `2` output classes.

Then, create a `4`-layer linear neural network with strictly more than `120` parameters, using the same number of features and classes.

Output the number of their parameters. You could also call `print` on the model objects themselves to see what output PyTorch gives you.

**Acceptance criteria:**

1. The number of parameters is calculated automatically.
2. The number of parameters corresponds to what is required in the description.

**Test case:**

Due to differences in architecture, your output may be different.

```python
python task07.py
```

```console
Number of parameters in network 1: 96
Number of parameters in network 2: 564
```

## Task 8

**Description:**

In this exercise, your goal is to find the optimal momentum and learning rate such that the optimizer can find the minimum of the following non-convex function in `20` steps:

$$x^{4} + x^{3} - 5x^{2}$$

Use the below code to create the plot seen in `notes.md`:

```python
def function(x):
    return x**4 + x**3 - 5 * x**2


def optimize_and_plot(lr, momentum):
    if lr > 0.05:
        raise ValueError('Choose a learning <= 0.05')
    x = torch.tensor(2.0, requires_grad=True)
    buffer = torch.zeros_like(x.data)
    values = []
    for i in range(20):

        y = function(x)
        values.append((x.clone(), y.clone()))
        y.backward()

        d_p = x.grad.data
        if momentum != 0:
            buffer.mul_(momentum).add_(d_p)
            d_p = buffer

        x.data.add_(d_p, alpha=-lr)
        x.grad.zero_()

    x = np.arange(-3, 2, 0.001)
    y = function(x)

    plt.figure(figsize=(10, 5))
    plt.plot([v[0].item() for v in values], [v[1].item() for v in values],
             'r-X',
             linewidth=2,
             markersize=7)
    for i in range(20):
        plt.text(values[i][0].item() + 0.1,
                 values[i][1].item(),
                 f'step {i}',
                 fontdict={'color': 'r'})
    plt.plot(x, y, linewidth=2)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(['Optimizer steps', 'Square function'])
    plt.tight_layout()
    plt.show()
```

**Acceptance criteria:**

1. The optimizer reaches the global minimum in `20` steps.

**Test case:**

Due to differences in numbers you choose, your output may be different.

```python
python task08.py
```

The following plot is produced:

![w03_task8.png](../assets/w03_task8.png "w03_task8.png")

## Task 9

**Description:**

Define a model with `4` linear layers. Initialize the values for their weights using the uniform distribution and freeze the first two layers. Output the weights and biases for the first five neurons in all layers.

**Acceptance criteria:**

1. Weights and biases in the first two layers are frozen.
2. All parameters are initialized using the uniform distribution.

**Test case:**

Due to randomness while initializing, your output may be different.

```python
python task09.py
```

```console


Layer 1:
tensor([[0.0482, 0.4436, 0.7622, 0.1222, 0.3652],
        [0.6463, 0.0278, 0.2894, 0.6294, 0.8911],
        [0.0176, 0.3099, 0.5541, 0.1009, 0.2061],
        [0.1273, 0.7479, 0.5153, 0.8127, 0.6082],
        [0.2606, 0.6177, 0.9061, 0.4381, 0.6572]])
tensor([0.6819, 0.6984, 0.1320, 0.7392, 0.4694])


Layer 2:
tensor([[0.9341, 0.6058, 0.4879, 0.8813, 0.3122, 0.2412, 0.3614, 0.0746, 0.1224,
         0.8624, 0.8382, 0.4743, 0.2846, 0.5055, 0.7221, 0.9662, 0.2481, 0.6267,
         0.5544, 0.2158, 0.8128, 0.3830, 0.7448, 0.8685, 0.9919, 0.1789, 0.7936,
         0.2630, 0.7048, 0.0556, 0.8059, 0.9410, 0.8612, 0.5677, 0.4717, 0.2409,
         0.0434, 0.5058, 0.3802, 0.4499, 0.6111, 0.8456, 0.3062, 0.9965, 0.7051,
         0.4636, 0.1359, 0.1839, 0.5725, 0.0103],
        [0.3350, 0.4217, 0.0320, 0.7187, 0.8875, 0.1440, 0.1477, 0.1812, 0.7518,
         0.7207, 0.4714, 0.3986, 0.9717, 0.9691, 0.2265, 0.6549, 0.2533, 0.2260,
         0.0795, 0.7555, 0.9385, 0.8391, 0.3973, 0.2928, 0.4406, 0.4022, 0.2924,
         0.5485, 0.6808, 0.1320, 0.4494, 0.7710, 0.6626, 0.5125, 0.9744, 0.2615,
         0.7686, 0.9259, 0.2670, 0.8346, 0.9211, 0.7656, 0.2541, 0.7519, 0.5865,
         0.0064, 0.6834, 0.0662, 0.2820, 0.1570]])
tensor([0.7046, 0.1667])


Layer 3:
tensor([[0.8046, 0.9029],
        [0.1294, 0.4815]], grad_fn=<SliceBackward0>)
tensor([0.2145, 0.4069], grad_fn=<SliceBackward0>)


Layer 4:
tensor([[0.1230, 0.7700]], grad_fn=<SliceBackward0>)
tensor([0.5153], grad_fn=<SliceBackward0>)
```

## Task 10

**Description:**

Create and train a neural network to predict whether a water sample is potable or drinkable (`1` or `0`) based on its chemical characteristics. Use the water potability dataset we discussed in class. It's present as two CSV files in our `DATA` folder as `water_train.csv` and `water_test.csv`. Output the distribution of the target values.

Perform some exploratory data analysis to choose the best features to use.

Play around with:

- The architecture: number of type of layers.
- The number of epochs.
- The batch size.
- The learning rate.
- The choice of optimizer.

Plot the average training and validation binary cross entropy losses per epoch as well as the metric you decide to use. Use `tqdm` to report progress through batches.

**Acceptance criteria:**

1. The `DataLoader` class is used.
2. The distribution of the target values is displayed in all sets.
3. The metric and loss on the test set are outputted.

**Test case:**

The below outputs were obtained using a test model - the output should show you the structure of the output. Due to differences in architecture, your output may be different.

```python
python task10.py
```

```console
Distribution of target values in training set:
            count  proportion
Potability                   
0             903    0.598806
1             605    0.401194
Distribution of target values in validation set:
            count  proportion
Potability                   
0             148    0.589641
1             103    0.410359
Distribution of target values in testing set:
            count  proportion
Potability                   
0             149     0.59127
1             103     0.40873
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 316.15it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 563.38it/s]
Epoch [1/30]:
 Average training loss: 0.6876169894107436
 Average validation loss: 0.6819964721798897
 Training metric score: 0.08708272874355316
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 338.15it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 563.62it/s]
Epoch [2/30]:
 Average training loss: 0.6860317250408193
 Average validation loss: 0.6771711651235819
 Training metric score: 0.12222222238779068
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 332.69it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 562.71it/s]
Epoch [3/30]:
 Average training loss: 0.6846164050240996
 Average validation loss: 0.6772961094975471
 Training metric score: 0.10511363297700882
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 336.04it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 561.34it/s]
Epoch [4/30]:
 Average training loss: 0.6865687280420273
 Average validation loss: 0.6803515665233135
 Training metric score: 0.09024745225906372
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 336.92it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 557.25it/s]
Epoch [5/30]:
 Average training loss: 0.6787174314102798
 Average validation loss: 0.6767929717898369
 Training metric score: 0.0
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 332.88it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 542.12it/s]
Epoch [6/30]:
 Average training loss: 0.6848205859383578
 Average validation loss: 0.6808604467660189
 Training metric score: 0.01610305905342102
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 335.27it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 561.85it/s]
Epoch [7/30]:
 Average training loss: 0.6799283302019513
 Average validation loss: 0.6815411597490311
 Training metric score: 0.0
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 336.89it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 563.21it/s]
Epoch [8/30]:
 Average training loss: 0.6821028503475997
 Average validation loss: 0.6832880824804306
 Training metric score: 0.08454810827970505
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 336.53it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 564.39it/s]
Epoch [9/30]:
 Average training loss: 0.677679481802794
 Average validation loss: 0.6833021156489849
 Training metric score: 0.16971279680728912
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 335.75it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 552.26it/s]
Epoch [10/30]:
 Average training loss: 0.6873839119124034
 Average validation loss: 0.6821667207404971
 Training metric score: 0.009771986864507198
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 335.20it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 561.82it/s]
Epoch [11/30]:
 Average training loss: 0.6784620871619572
 Average validation loss: 0.6819236818701029
 Training metric score: 0.0
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 334.71it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 562.39it/s]
Epoch [12/30]:
 Average training loss: 0.6898293894119364
 Average validation loss: 0.6804907843470573
 Training metric score: 0.13351134955883026
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 334.64it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 558.41it/s]
Epoch [13/30]:
 Average training loss: 0.6786409559704009
 Average validation loss: 0.690683638677001
 Training metric score: 0.08875739574432373
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 336.85it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 555.08it/s]
Epoch [14/30]:
 Average training loss: 0.6771163394842198
 Average validation loss: 0.6778231244534254
 Training metric score: 0.06656581163406372
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 335.80it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 563.59it/s]
Epoch [15/30]:
 Average training loss: 0.6819966145293422
 Average validation loss: 0.6800849344581366
 Training metric score: 0.0065146577544510365
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 337.14it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 562.92it/s]
Epoch [16/30]:
 Average training loss: 0.684752968884019
 Average validation loss: 0.6781640015542507
 Training metric score: 0.12947657704353333
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 336.27it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 558.15it/s]
Epoch [17/30]:
 Average training loss: 0.6831075388287741
 Average validation loss: 0.7014249172061682
 Training metric score: 0.10857142508029938
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 336.49it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 560.57it/s]
Epoch [18/30]:
 Average training loss: 0.6786818994731499
 Average validation loss: 0.6750532742589712
 Training metric score: 0.0
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 333.18it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 560.44it/s]
Epoch [19/30]:
 Average training loss: 0.6805883250223896
 Average validation loss: 0.6935788653790951
 Training metric score: 0.05246913433074951
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 336.04it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 563.94it/s]
Epoch [20/30]:
 Average training loss: 0.6849787574281138
 Average validation loss: 0.6811317577958107
 Training metric score: 0.0773809552192688
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 335.02it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 560.68it/s]
Epoch [21/30]:
 Average training loss: 0.6799326857561787
 Average validation loss: 0.7111492604017258
 Training metric score: 0.1311018168926239
 Validation metric score: 0.5819209218025208
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 336.52it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 551.37it/s]
Epoch [22/30]:
 Average training loss: 0.6824536514345301
 Average validation loss: 0.6766159012913704
 Training metric score: 0.07726597040891647
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 333.09it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 559.42it/s]
Epoch [23/30]:
 Average training loss: 0.6802576898582398
 Average validation loss: 0.6816935483366251
 Training metric score: 0.012987012974917889
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 333.73it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 542.68it/s]
Epoch [24/30]:
 Average training loss: 0.6833551931318151
 Average validation loss: 0.6817533746361732
 Training metric score: 0.02239999920129776
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 335.64it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 562.46it/s]
Epoch [25/30]:
 Average training loss: 0.6805756117301013
 Average validation loss: 0.6765081714838743
 Training metric score: 0.06646525859832764
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 336.69it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 553.02it/s]
Epoch [26/30]:
 Average training loss: 0.6820322458075467
 Average validation loss: 0.6832163594663143
 Training metric score: 0.009819967672228813
 Validation metric score: 0.01923076994717121
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 335.47it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 559.13it/s]
Epoch [27/30]:
 Average training loss: 0.6794978630921197
 Average validation loss: 0.6818940434604883
 Training metric score: 0.019323671236634254
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 335.56it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 559.00it/s]
Epoch [28/30]:
 Average training loss: 0.6808573229918404
 Average validation loss: 0.6799554508179426
 Training metric score: 0.05246913433074951
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 335.95it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 555.87it/s]
Epoch [29/30]:
 Average training loss: 0.6829377307147576
 Average validation loss: 0.6768536306917667
 Training metric score: 0.052877139300107956
 Validation metric score: 0.0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 336.44it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 556.91it/s]
Epoch [30/30]:
 Average training loss: 0.683645261973931
 Average validation loss: 0.6792476829141378
 Training metric score: 0.15796519815921783
 Validation metric score: 0.0
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 559.80it/s]
Average test loss: 0.67628463357687
Test metric score: 0.0
```

And the following plot is produced:

![w03_task10_result.png](../assets/w03_task10_result.png "w03_task10_result.png")

## Task 11

**Description:**

Create a neural network with a linear layer that squashes its input to `16` dimensions, a ReLU layer, and a dropout layer. During training the network has to be able to randomly disable `80%` of its neurons. Pass `input_tensor` to the model and output the result.

```python
input_tensor = torch.tensor([[0.1819184124469757, 0.16495776176452637, 0.15381167829036713, 0.27550166845321655, 0.2244979590177536, 0.1901470422744751, 0.3754608631134033, 0.2909483015537262, 0.24059511721134186, 0.45058903098106384, 0.35181766748428345, 0.2998202443122864, 0.49866142868995667, 0.3827182352542877, 0.3141459822654724, 0.5220781564712524, 0.39985108375549316, 0.32656028866767883, 0.5345920920372009, 0.412087619304657, 0.3374066650867462, 0.5353729724884033, 0.4120948910713196, 0.3435279428958893, 0.5176263451576233, 0.3971695899963379, 0.3265799582004547, 0.5365544557571411, 0.4157505929470062, 0.3459533751010895, 0.5698273181915283, 0.4494820237159729, 0.383587509393692, 0.5887288451194763, 0.47119006514549255, 0.4198046028614044, 0.6100356578826904, 0.5030798316001892, 0.45634421706199646, 0.6262279152870178, 0.5265932679176331, 0.4789578914642334, 0.6344038248062134, 0.5242546200752258, 0.48088759183883667, 0.6329152584075928, 0.5253323912620544, 0.4833231270313263, 0.6327783465385437, 0.5291191935539246, 0.4879509508609772, 0.6373038291931152, 0.5347554683685303, 0.49849632382392883, 0.6376794576644897, 0.5285146832466125, 0.48574551939964294, 0.628280520439148, 0.5124699473381042, 0.471763014793396, 0.6253150105476379, 0.5196305513381958, 0.47051510214805603, 0.6432173252105713, 0.5330312848091125, 0.4945002794265747, 0.6664242744445801, 0.5713988542556763, 0.5230433940887451, 0.6853389143943787, 0.5904213786125183, 0.5427646040916443, 0.6978332996368408, 0.6065328121185303, 0.5568618178367615, 0.6943079829216003, 0.6024137139320374, 0.554010272026062, 0.6788963079452515, 0.5893381834030151, 0.5414921045303345, 0.6729124784469604, 0.5789656639099121, 0.5293691158294678, 0.6194267868995667, 0.5104096531867981, 0.4455663561820984, 0.5725225210189819, 0.44765910506248474, 0.3773389756679535, 0.38689297437667847, 0.3148936927318573, 0.2855091691017151, 0.12450283765792847, 0.11579212546348572, 0.12389758974313736, 0.23127028346061707, 0.19283288717269897, 0.18055933713912964, 0.36345934867858887, 0.28031623363494873, 0.2401789128780365, 0.45103055238723755, 0.34353262186050415, 0.2856754958629608, 0.498017281293869, 0.38707244396209717, 0.3240499198436737, 0.5233540534973145, 0.40325894951820374, 0.3322470784187317, 0.5290525555610657, 0.4071119725704193, 0.33371010422706604, 0.5337569713592529, 0.41166219115257263, 0.34036698937416077, 0.5450513958930969, 0.4212748408317566, 0.35306793451309204, 0.5560584664344788, 0.43985822796821594, 0.3669014573097229, 0.5797246098518372, 0.46242260932922363, 0.4007290005683899, 0.5974323153495789, 0.4842345118522644, 0.4432956874370575, 0.634335994720459, 0.5238890647888184, 0.4939703643321991, 0.6336001753807068, 0.5436570048332214, 0.5171237587928772, 0.6357924938201904, 0.5423905253410339, 0.5226799845695496, 0.6532943248748779, 0.5585663914680481, 0.5302010178565979, 0.6606267094612122, 0.5568056106567383, 0.5311674475669861, 0.6496113538742065, 0.5534089803695679, 0.5150609612464905, 0.6423532366752625, 0.5362401008605957, 0.5031194686889648, 0.6389397382736206, 0.5320115685462952, 0.4895516037940979, 0.6185283064842224, 0.4994184374809265, 0.463503360748291, 0.6161976456642151, 0.4958425462245941, 0.46373018622398376, 0.6273001432418823, 0.5179946422576904, 0.48698243498802185, 0.6493526697158813, 0.5455436706542969, 0.4976364076137543, 0.6565961837768555, 0.5572801232337952, 0.5128807425498962, 0.662687361240387, 0.5690248012542725, 0.5341235399246216, 0.6593703627586365, 0.5751903057098389, 0.5459816455841064, 0.6105800867080688, 0.5418475270271301, 0.5177072882652283, 0.5323490500450134, 0.47640135884284973, 0.4641609489917755, 0.5131914019584656, 0.4471403658390045, 0.42065680027008057, 0.5528342723846436, 0.4481481909751892, 0.40319082140922546, 0.4262270927429199, 0.35231417417526245, 0.3212384879589081, 0.12076519429683685, 0.11182095110416412, 0.1242021694779396, 0.24036680161952972, 0.2072068154811859, 0.18707653880119324, 0.39241543412208557, 0.30686262249946594, 0.2558879852294922, 0.4757629930973053, 0.36610278487205505, 0.29841700196266174, 0.5170558094978333, 0.4005037546157837, 0.33077019453048706, 0.5340683460235596, 0.4103721082210541, 0.3489653766155243, 0.5397830605506897, 0.41176173090934753, 0.3483563959598541, 0.5305297374725342, 0.40614309906959534, 0.34118548035621643, 0.5358593463897705, 0.41120201349258423, 0.3503001034259796, 0.5587635636329651, 0.45049726963043213, 0.3885449171066284, 0.5353791117668152, 0.43871843814849854, 0.4062241315841675, 0.48465001583099365, 0.42028576135635376, 0.394135981798172, 0.4466187059879303, 0.366423636674881, 0.3461426794528961, 0.49052757024765015, 0.4259488582611084, 0.4090404808521271, 0.5428964495658875, 0.4843341112136841, 0.46806827187538147, 0.593441903591156, 0.5053554773330688, 0.490709125995636, 0.6102200746536255, 0.5145905017852783, 0.498226523399353, 0.5926057696342468, 0.48895856738090515, 0.46360841393470764, 0.5765431523323059, 0.455993115901947, 0.4309294521808624, 0.5994648337364197, 0.49011102318763733, 0.44819727540016174, 0.6190520524978638, 0.5035994052886963, 0.4651733934879303, 0.6104128956794739, 0.48361602425575256, 0.4529260993003845, 0.5763982534408569, 0.4654330611228943, 0.4308757185935974, 0.5670548677444458, 0.4554111659526825, 0.4132518470287323, 0.5524818897247314, 0.45042040944099426, 0.4154594838619232, 0.5407674908638, 0.4624878168106079, 0.43765324354171753, 0.47136402130126953, 0.41903039813041687, 0.4013965427875519, 0.359292209148407, 0.327382355928421, 0.3212616443634033, 0.27333763241767883, 0.2507430613040924, 0.2613213360309601, 0.3064744770526886, 0.28022676706314087, 0.275942325592041, 0.4146658480167389, 0.3506481945514679, 0.3347182273864746, 0.4182860851287842, 0.3487866520881653, 0.315464586019516, 0.1021222323179245, 0.09997398406267166, 0.10761863738298416, 0.2430541068315506, 0.2094191610813141, 0.19137980043888092, 0.4129762649536133, 0.3218853771686554, 0.2647893726825714, 0.49665653705596924, 0.3825640380382538, 0.3194386065006256, 0.5315473675727844, 0.41390037536621094, 0.34717318415641785, 0.5410330295562744, 0.41959333419799805, 0.3568437695503235, 0.5359730124473572, 0.411831796169281, 0.34523677825927734, 0.5364254713058472, 0.4180298149585724, 0.3610841631889343, 0.5127047896385193, 0.4063106179237366, 0.36381107568740845, 0.444903165102005, 0.37457358837127686, 0.3404417932033539, 0.3654930591583252, 0.31625896692276, 0.30483484268188477, 0.2758166790008545, 0.25074928998947144, 0.24695473909378052, 0.23951204121112823, 0.21885140240192413, 0.22190076112747192, 0.2698364853858948, 0.24948620796203613, 0.24183453619480133, 0.33360984921455383, 0.3024561405181885, 0.29073667526245117, 0.4001275300979614, 0.3482084572315216, 0.3413901925086975, 0.437240332365036, 0.37634724378585815, 0.36426571011543274, 0.4560149610042572, 0.37535515427589417, 0.35965049266815186, 0.4563280940055847, 0.36171838641166687, 0.3459022343158722, 0.5203631520271301, 0.4139029383659363, 0.3838968575000763, 0.5721607208251953, 0.46027040481567383, 0.4197174906730652, 0.5700844526290894, 0.4558655917644501, 0.42071130871772766, 0.5355936288833618, 0.4202023446559906, 0.3848600685596466, 0.4970855414867401, 0.3893239498138428, 0.35699766874313354, 0.41121140122413635, 0.3330765664577484, 0.32004231214523315, 0.2843729257583618, 0.2482082098722458, 0.24090443551540375, 0.20011812448501587, 0.18432079255580902, 0.18137016892433167, 0.16804184019565582, 0.1582757979631424, 0.1575489342212677, 0.17487096786499023, 0.162826269865036, 0.15548989176750183, 0.2212904691696167, 0.1981482207775116, 0.1925855576992035, 0.291751891374588, 0.2669823169708252, 0.2521655261516571, 0.3829197585582733, 0.331583172082901, 0.29927897453308105, 0.15831492841243744, 0.14311841130256653, 0.15732280910015106, 0.2505756914615631, 0.2094595581293106, 0.19519470632076263, 0.41409850120544434, 0.3236384093761444, 0.2710501551628113, 0.49071788787841797, 0.3745458722114563, 0.3116280436515808, 0.5184863209724426, 0.3987601399421692, 0.33417263627052307, 0.5231500864028931, 0.4034205973148346, 0.3388361930847168, 0.513520359992981, 0.3966871500015259, 0.3323614001274109, 0.5000682473182678, 0.3973296880722046, 0.3484654128551483, 0.4204166829586029, 0.34503868222236633, 0.31552374362945557, 0.34055253863334656, 0.2875575125217438, 0.266323447227478, 0.28404703736305237, 0.24725967645645142, 0.23471450805664062, 0.23681044578552246, 0.20552773773670197, 0.2049141526222229, 0.20113277435302734, 0.17985834181308746, 0.18144184350967407, 0.18196409940719604, 0.1673617660999298, 0.16905006766319275, 0.17273567616939545, 0.15388689935207367, 0.15547683835029602, 0.18787698447704315, 0.1668558418750763, 0.16389654576778412, 0.21895310282707214, 0.1858094185590744, 0.1777108758687973, 0.3099452257156372, 0.25477829575538635, 0.24195609986782074, 0.33554574847221375, 0.26467689871788025, 0.24899178743362427, 0.3916759490966797, 0.30346229672431946, 0.28448277711868286, 0.5024822950363159, 0.3929142951965332, 0.35773715376853943, 0.5050311088562012, 0.39925599098205566, 0.369478315114975, 0.44566810131073, 0.3544788956642151, 0.3223537802696228, 0.3679869472980499, 0.29602909088134766, 0.26592373847961426, 0.26255813241004944, 0.2118895947933197, 0.19514022767543793, 0.19778096675872803, 0.166486918926239, 0.14938130974769592, 0.20330321788787842, 0.17402514815330505, 0.15217170119285583, 0.23479297757148743, 0.19849315285682678, 0.1819596290588379, 0.2525664269924164, 0.20807597041130066, 0.19617074728012085, 0.27787014842033386, 0.2193024754524231, 0.2093522697687149, 0.32726991176605225, 0.2767467200756073, 0.253607839345932, 0.4012608230113983, 0.3349958062171936, 0.30779650807380676, 0.2109285593032837, 0.19423992931842804, 0.20950102806091309, 0.2537853419780731, 0.21460266411304474, 0.18998606503009796, 0.4239092171192169, 0.3339553773403168, 0.27102452516555786, 0.49048912525177, 0.3764584958553314, 0.31297484040260315, 0.4973372519016266, 0.38023272156715393, 0.31339871883392334, 0.5007650256156921, 0.3838474750518799, 0.31718653440475464, 0.4960564374923706, 0.38806599378585815, 0.3276219367980957, 0.44979527592658997, 0.35818609595298767, 0.3104560673236847, 0.39479440450668335, 0.30707693099975586, 0.2814553678035736, 0.3371952772140503, 0.2689375877380371, 0.25176599621772766, 0.2851160168647766, 0.23222726583480835, 0.21404238045215607, 0.25806501507759094, 0.20741362869739532, 0.1944708526134491, 0.2242363691329956, 0.18251235783100128, 0.16893896460533142, 0.2142878770828247, 0.1769329309463501, 0.16463255882263184, 0.213671013712883, 0.17470233142375946, 0.15899230539798737, 0.21453909575939178, 0.16820190846920013, 0.1473662108182907, 0.22437292337417603, 0.1755029261112213, 0.1577635109424591, 0.253792941570282, 0.20022520422935486, 0.18016932904720306, 0.2681542932987213, 0.20209883153438568, 0.18664376437664032, 0.30497217178344727, 0.23323467373847961, 0.21643836796283722, 0.42346519231796265, 0.32341423630714417, 0.2953382432460785, 0.432962566614151, 0.3273492753505707, 0.29560181498527527, 0.3249220550060272, 0.243421733379364, 0.20835652947425842, 0.26728662848472595, 0.2047434151172638, 0.17905119061470032, 0.28885510563850403, 0.2172035127878189, 0.19420567154884338, 0.345100998878479, 0.26757577061653137, 0.23411744832992554, 0.4039124548435211, 0.32202011346817017, 0.28411760926246643, 0.43572065234184265, 0.3452858030796051, 0.31204429268836975, 0.4175652265548706, 0.33465147018432617, 0.29888400435447693, 0.3777616322040558, 0.2991703450679779, 0.2594279646873474, 0.3857804238796234, 0.3247547745704651, 0.26811686158180237, 0.44467246532440186, 0.3695654273033142, 0.33281782269477844, 0.12501345574855804, 0.10783617943525314, 0.11941663920879364, 0.3321942687034607, 0.28723013401031494, 0.2643490135669708, 0.4441787004470825, 0.362657368183136, 0.3195529878139496, 0.4860151410102844, 0.36729127168655396, 0.3082443177700043, 0.48650842905044556, 0.3764675557613373, 0.3127945363521576, 0.48269176483154297, 0.37557318806648254, 0.3064206540584564, 0.45668110251426697, 0.3619982898235321, 0.3016434609889984, 0.39823368191719055, 0.3111323118209839, 0.26501867175102234, 0.360377699136734, 0.2656936049461365, 0.23250223696231842, 0.33659249544143677, 0.2514221668243408, 0.22114253044128418, 0.32179883122444153, 0.23855651915073395, 0.21412774920463562, 0.33109596371650696, 0.25048401951789856, 0.2189878225326538, 0.3596287667751312, 0.2823972702026367, 0.2462235689163208, 0.3947254717350006, 0.31107214093208313, 0.2820579707622528, 0.40245190262794495, 0.3168184161186218, 0.289012610912323, 0.37853676080703735, 0.2928626537322998, 0.2575821280479431, 0.3085656464099884, 0.22578684985637665, 0.19205719232559204, 0.2409287542104721, 0.17333607375621796, 0.15096025168895721, 0.2555127739906311, 0.18807056546211243, 0.16151221096515656, 0.3099793791770935, 0.2328767031431198, 0.20995555818080902, 0.40525755286216736, 0.3054969012737274, 0.27267423272132874, 0.4371654689311981, 0.33503273129463196, 0.2947593927383423, 0.32768797874450684, 0.24096600711345673, 0.21064400672912598, 0.29780158400535583, 0.21260954439640045, 0.18553687632083893, 0.3615272641181946, 0.2706921100616455, 0.24359288811683655, 0.4252921938896179, 0.32601967453956604, 0.294251948595047, 0.47242674231529236, 0.37481486797332764, 0.3404806852340698, 0.5018481612205505, 0.4110713005065918, 0.37297147512435913, 0.47902894020080566, 0.3996027708053589, 0.3680191934108734, 0.4065670967102051, 0.34638574719429016, 0.3152335286140442, 0.418372243642807, 0.37040165066719055, 0.33668509125709534, 0.4494536817073822, 0.40262770652770996, 0.3890780806541443, 0.3262045383453369, 0.31400266289711, 0.317305326461792, 0.40738818049430847, 0.32975053787231445, 0.29475465416908264, 0.4215531647205353, 0.35148584842681885, 0.3164392113685608, 0.45908647775650024, 0.35918229818344116, 0.32878103852272034, 0.48292824625968933, 0.37742313742637634, 0.32521483302116394, 0.4795643389225006, 0.37377968430519104, 0.30917221307754517, 0.42887482047080994, 0.335841566324234, 0.28070807456970215, 0.3735974431037903, 0.2867133617401123, 0.24162350594997406, 0.33645978569984436, 0.2575186491012573, 0.2143934667110443, 0.3352457880973816, 0.2550770044326782, 0.21256907284259796, 0.3600616753101349, 0.26986563205718994, 0.23053349554538727, 0.385664701461792, 0.29547926783561707, 0.2536027729511261, 0.4002173840999603, 0.3081951141357422, 0.2699415385723114, 0.43307745456695557, 0.33868446946144104, 0.3019627332687378, 0.45026007294654846, 0.3519550859928131, 0.31482309103012085, 0.43926459550857544, 0.3368002474308014, 0.3022894859313965, 0.34359630942344666, 0.25018399953842163, 0.2225378304719925, 0.26022374629974365, 0.18449102342128754, 0.1614433228969574, 0.30931711196899414, 0.22234326601028442, 0.1937296837568283, 0.3689733147621155, 0.27171558141708374, 0.23904046416282654, 0.4981820285320282, 0.3909512162208557, 0.3530195653438568, 0.5341639518737793, 0.42433398962020874, 0.3788283169269562, 0.3614392578601837, 0.2692912817001343, 0.23330099880695343, 0.27936843037605286, 0.1907803863286972, 0.1596275418996811, 0.3077537715435028, 0.2184550166130066, 0.19737140834331512, 0.3685442805290222, 0.2867100238800049, 0.26466456055641174, 0.4023454487323761, 0.3236350119113922, 0.30349627137184143, 0.4253351390361786, 0.36456573009490967, 0.34057167172431946, 0.4279176592826843, 0.3790149390697479, 0.3500073552131653, 0.4344378709793091, 0.3777761459350586, 0.35080358386039734, 0.4238153398036957, 0.3623179793357849, 0.3322553038597107, 0.2806655466556549, 0.24292990565299988, 0.24190670251846313, 0.36226019263267517, 0.33893126249313354, 0.33441534638404846, 0.44139233231544495, 0.3406789302825928, 0.291049987077713, 0.443315714597702, 0.34553882479667664, 0.29625558853149414, 0.4277222156524658, 0.35400334000587463, 0.3048044443130493, 0.4786008596420288, 0.3931915760040283, 0.35358983278274536, 0.4825112819671631, 0.3797174096107483, 0.3216351866722107, 0.46261414885520935, 0.373869389295578, 0.31568434834480286, 0.4216358959674835, 0.34084129333496094, 0.2952689826488495, 0.36772027611732483, 0.28133276104927063, 0.2403961569070816, 0.32252904772758484, 0.23714324831962585, 0.19463789463043213, 0.31753453612327576, 0.22529010474681854, 0.19254933297634125, 0.3094753623008728, 0.22965091466903687, 0.20421665906906128, 0.3267107605934143, 0.24696047604084015, 0.22077623009681702, 0.3565449118614197, 0.2715754508972168, 0.24386733770370483, 0.36263537406921387, 0.2843499481678009, 0.25394096970558167, 0.31877321004867554, 0.2510222792625427, 0.22360192239284515, 0.264102965593338, 0.2026216685771942, 0.1780555099248886, 0.31225866079330444, 0.24675697088241577, 0.21694588661193848, 0.36337408423423767, 0.2774771451950073, 0.23569346964359283, 0.4077127277851105, 0.30799034237861633, 0.2664480209350586, 0.5645113587379456, 0.46520689129829407, 0.41630151867866516, 0.6017525792121887, 0.5011811256408691, 0.44337964057922363, 0.44075778126716614, 0.34089377522468567, 0.30003273487091064, 0.31334128975868225, 0.26534369587898254, 0.2414584457874298, 0.30411919951438904, 0.26074832677841187, 0.2483532428741455, 0.2572638988494873, 0.20599490404129028, 0.20362405478954315, 0.24432307481765747, 0.1881704181432724, 0.18140676617622375, 0.2815861105918884, 0.2276306003332138, 0.2121226042509079, 0.3982822000980377, 0.3397482931613922, 0.3101407289505005, 0.5334187150001526, 0.44941720366477966, 0.3983671963214874, 0.4393884837627411, 0.3694537878036499, 0.33652442693710327, 0.23950736224651337, 0.20999763906002045, 0.22300052642822266, 0.3666420876979828, 0.3459809720516205, 0.331903874874115, 0.4477894902229309, 0.34184008836746216, 0.2884584069252014, 0.4539611339569092, 0.3463212847709656, 0.29230305552482605, 0.4356555938720703, 0.3431258499622345, 0.2855152487754822, 0.44869959354400635, 0.3782191574573517, 0.3213001787662506, 0.4584640860557556, 0.38052067160606384, 0.3409044146537781, 0.48134616017341614, 0.3934508264064789, 0.33948254585266113, 0.5019522309303284, 0.4059489667415619, 0.356557160615921, 0.47667375206947327, 0.3839390277862549, 0.346843421459198, 0.39891791343688965, 0.3380897343158722, 0.3066129982471466, 0.36722004413604736, 0.3130534589290619, 0.29301318526268005, 0.33200737833976746, 0.2837141454219818, 0.2694648802280426, 0.3007216155529022, 0.2560819983482361, 0.24677108228206635, 0.2776295840740204, 0.2341618686914444, 0.22799049317836761, 0.26880040764808655, 0.22644737362861633, 0.2235763818025589, 0.282516747713089, 0.2379409372806549, 0.22433806955814362, 0.32005152106285095, 0.26710718870162964, 0.24843639135360718, 0.3586258292198181, 0.29270899295806885, 0.27383294701576233, 0.3880745768547058, 0.3280274569988251, 0.29718977212905884, 0.3996746242046356, 0.33203208446502686, 0.30421891808509827, 0.5423213839530945, 0.4623144567012787, 0.43446922302246094, 0.5934891700744629, 0.5116329193115234, 0.48309415578842163, 0.487643301486969, 0.4167160987854004, 0.3927569091320038, 0.38537999987602234, 0.32880914211273193, 0.3187355101108551, 0.3758230209350586, 0.3039400279521942, 0.27819618582725525, 0.3572472333908081, 0.2779666781425476, 0.2539912462234497, 0.3592946529388428, 0.2876226305961609, 0.2585418224334717, 0.41653549671173096, 0.34535613656044006, 0.3025706112384796, 0.5172079205513, 0.43685561418533325, 0.3898264467716217, 0.5897716879844666, 0.4957767426967621, 0.4456777274608612, 0.4399482011795044, 0.36910295486450195, 0.3427652418613434, 0.28925901651382446, 0.2728920578956604, 0.2702329158782959, 0.403397798538208, 0.3821486532688141, 0.3733486831188202, 0.44934192299842834, 0.34452179074287415, 0.2921111285686493, 0.44609594345092773, 0.3437613546848297, 0.2842026650905609, 0.45372849702835083, 0.3521726727485657, 0.29005730152130127, 0.45791512727737427, 0.3595571517944336, 0.2943621277809143, 0.46711331605911255, 0.3681195378303528, 0.3218625485897064, 0.43636631965637207, 0.36206918954849243, 0.3223018944263458, 0.44810470938682556, 0.4004652500152588, 0.36018136143684387, 0.45758455991744995, 0.39505699276924133, 0.3792518973350525, 0.481640487909317, 0.4019168019294739, 0.3790028989315033, 0.4441533088684082, 0.356738805770874, 0.33395496010780334, 0.43648988008499146, 0.34819144010543823, 0.3249630630016327, 0.4329581558704376, 0.34953150153160095, 0.31928694248199463, 0.422965943813324, 0.33574798703193665, 0.30769336223602295, 0.41994795203208923, 0.33239874243736267, 0.3067615032196045, 0.38815274834632874, 0.30433979630470276, 0.2689454257488251, 0.40837275981903076, 0.3135257661342621, 0.27943935990333557, 0.45821139216423035, 0.35985520482063293, 0.32608911395072937, 0.4137345850467682, 0.3364495635032654, 0.31453588604927063, 0.37982282042503357, 0.3271180987358093, 0.3047757148742676, 0.4804426431655884, 0.409592866897583, 0.37971749901771545, 0.5576796531677246, 0.471416175365448, 0.4345768988132477, 0.4328119456768036, 0.34892162680625916, 0.32705211639404297, 0.35741496086120605, 0.3084737956523895, 0.2974807322025299, 0.47677886486053467, 0.37753617763519287, 0.3402496576309204, 0.4105537533760071, 0.3087531626224518, 0.27603617310523987, 0.4027385413646698, 0.2942143678665161, 0.2587941884994507, 0.459034264087677, 0.3510783314704895, 0.3206048607826233, 0.5393195152282715, 0.434520423412323, 0.3979896903038025, 0.5734565854072571, 0.4736328721046448, 0.4318661391735077, 0.41576048731803894, 0.3387335240840912, 0.32940787076950073, 0.44903698563575745, 0.4503813087940216, 0.4711766242980957, 0.5999010801315308, 0.5780518054962158, 0.5736359357833862, 0.44642913341522217, 0.33977723121643066, 0.2926277220249176, 0.45587924122810364, 0.3551217019557953, 0.2938748598098755, 0.46646061539649963, 0.36670294404029846, 0.29503133893013, 0.4708245098590851, 0.37156209349632263, 0.3069861829280853, 0.486395925283432, 0.37305277585983276, 0.31406325101852417, 0.4864899814128876, 0.3886776864528656, 0.33132898807525635, 0.48126694560050964, 0.3951859772205353, 0.34592902660369873, 0.42147761583328247, 0.34546080231666565, 0.315775066614151, 0.5275416970252991, 0.42978668212890625, 0.3971095383167267, 0.5334477424621582, 0.42264413833618164, 0.3966946601867676, 0.4817630350589752, 0.3729638457298279, 0.3437689542770386, 0.4408029019832611, 0.34104427695274353, 0.3056316077709198, 0.4315062165260315, 0.33530497550964355, 0.2961018681526184, 0.41073524951934814, 0.3164163827896118, 0.277218222618103, 0.43143442273139954, 0.3309500217437744, 0.29492437839508057, 0.46274569630622864, 0.3507969379425049, 0.3189716041088104, 0.4716047942638397, 0.35981228947639465, 0.3273194134235382, 0.3663279712200165, 0.2978176474571228, 0.28041303157806396, 0.33731546998023987, 0.2724294662475586, 0.24392367899417877, 0.4858015179634094, 0.39695534110069275, 0.35676220059394836, 0.6038300395011902, 0.4957275986671448, 0.4527156352996826, 0.5191516876220703, 0.4128098785877228, 0.37421542406082153, 0.4151882231235504, 0.3514731824398041, 0.3235495090484619, 0.5075838565826416, 0.3925181031227112, 0.36410120129585266, 0.4981895089149475, 0.3748500347137451, 0.35091662406921387, 0.46602943539619446, 0.3496999144554138, 0.317999929189682, 0.4695267081260681, 0.35885944962501526, 0.32186445593833923, 0.5019092559814453, 0.39623579382896423, 0.3665355145931244, 0.5216799378395081, 0.43826940655708313, 0.4297946095466614, 0.4611007571220398, 0.44720107316970825, 0.4651746153831482, 0.5753834843635559, 0.5833280086517334, 0.5952486991882324, 0.46272969245910645, 0.45043718814849854, 0.4709835648536682, 0.44578349590301514, 0.33684447407722473, 0.2928408682346344, 0.4624882936477661, 0.35910168290138245, 0.3070472478866577, 0.4669044017791748, 0.365293949842453, 0.30718696117401123, 0.4703252911567688, 0.37001633644104004, 0.31535208225250244, 0.45516982674598694, 0.34572601318359375, 0.2910754084587097, 0.457660436630249, 0.3533780574798584, 0.308671236038208, 0.4687647521495819, 0.3634966313838959, 0.3133581578731537, 0.4666774868965149, 0.36972036957740784, 0.3229711055755615, 0.4748767614364624, 0.37490254640579224, 0.3335954248905182, 0.5009987950325012, 0.39116373658180237, 0.3547188937664032, 0.5144674777984619, 0.4021293818950653, 0.36164531111717224, 0.46951574087142944, 0.3597123622894287, 0.32454684376716614, 0.44587504863739014, 0.34189483523368835, 0.3067229986190796, 0.45814868807792664, 0.35227465629577637, 0.32041627168655396, 0.48243752121925354, 0.3724801242351532, 0.3422752916812897, 0.4793992340564728, 0.35863009095191956, 0.333972305059433, 0.49699854850769043, 0.3793095648288727, 0.34622353315353394, 0.47402557730674744, 0.36672836542129517, 0.3300226032733917, 0.450443297624588, 0.36245015263557434, 0.32387205958366394, 0.526193380355835, 0.42325034737586975, 0.39021921157836914, 0.6228117942810059, 0.5163047313690186, 0.48124799132347107, 0.6034122109413147, 0.48931291699409485, 0.4440554082393646, 0.5378822684288025, 0.42553219199180603, 0.4035355746746063, 0.535576343536377, 0.413700133562088, 0.3927448093891144, 0.5274866819381714, 0.40070652961730957, 0.3744944632053375, 0.5177599191665649, 0.41531458497047424, 0.3851984739303589, 0.5066019296646118, 0.43067315220832825, 0.41517046093940735, 0.5572851300239563, 0.49391430616378784, 0.49348634481430054, 0.5764498114585876, 0.5179656147956848, 0.5152478814125061, 0.5520440340042114, 0.48108237981796265, 0.47326377034187317, 0.4109468162059784, 0.3588864505290985, 0.3523721396923065, 0.10481114685535431, 0.11070437729358673, 0.1267983317375183, 0.4432545304298401, 0.3348233699798584, 0.2908281087875366, 0.45405223965644836, 0.35049742460250854, 0.3034321367740631, 0.4607884883880615, 0.35867828130722046, 0.3113933503627777, 0.45894894003868103, 0.35539373755455017, 0.3072403371334076, 0.44277870655059814, 0.3392166197299957, 0.2921561598777771, 0.43689820170402527, 0.33398574590682983, 0.28947219252586365, 0.4343094825744629, 0.33099988102912903, 0.2859053909778595, 0.434753954410553, 0.33795198798179626, 0.3031717836856842, 0.4537702202796936, 0.36693862080574036, 0.3355109393596649, 0.42860931158065796, 0.32216551899909973, 0.29642754793167114, 0.43579116463661194, 0.3174552917480469, 0.2889120280742645, 0.4470188617706299, 0.3295745849609375, 0.29594919085502625, 0.4737001061439514, 0.362676739692688, 0.33021876215934753, 0.4925483167171478, 0.3863148093223572, 0.3490758240222931, 0.47205305099487305, 0.36557963490486145, 0.3295048177242279, 0.43529289960861206, 0.3170531690120697, 0.28615179657936096, 0.44795212149620056, 0.3323187530040741, 0.30622509121894836, 0.4851381778717041, 0.3741888999938965, 0.3450302481651306, 0.500346839427948, 0.37108686566352844, 0.3445081412792206, 0.5802413821220398, 0.4522017538547516, 0.42583751678466797, 0.6549162864685059, 0.533194363117218, 0.4974328875541687, 0.6573179960250854, 0.534687876701355, 0.4948788583278656, 0.550155758857727, 0.41927027702331543, 0.38486629724502563, 0.4283190071582794, 0.31167837977409363, 0.29060301184654236, 0.5076479315757751, 0.40955689549446106, 0.3874204158782959, 0.5445247292518616, 0.4497758150100708, 0.4262418746948242, 0.5481122732162476, 0.457851767539978, 0.4332680106163025, 0.5278770327568054, 0.42741885781288147, 0.4029429256916046, 0.5116263628005981, 0.39874517917633057, 0.3665238618850708, 0.4713960886001587, 0.3641965687274933, 0.3267333507537842, 0.3184721767902374, 0.26095330715179443, 0.25336161255836487, 0.08133262395858765, 0.09419353306293488, 0.10883205384016037, 0.43347588181495667, 0.3298199772834778, 0.2863045930862427, 0.4542388916015625, 0.350739449262619, 0.30151569843292236, 0.45907726883888245, 0.35674849152565, 0.3097469210624695, 0.46182599663734436, 0.35595184564590454, 0.30950728058815, 0.4405308663845062, 0.3346565365791321, 0.29141029715538025, 0.4283120036125183, 0.3218168020248413, 0.2832096517086029, 0.4108297824859619, 0.3054676353931427, 0.270162433385849, 0.4177568256855011, 0.3211260139942169, 0.2836384177207947, 0.5152903199195862, 0.4570591449737549, 0.4262499213218689, 0.5078681707382202, 0.4441450238227844, 0.4229721426963806, 0.47390294075012207, 0.39208653569221497, 0.36658844351768494, 0.48577880859375, 0.390836238861084, 0.3615126311779022, 0.47888967394828796, 0.3803039491176605, 0.35553914308547974, 0.47057241201400757, 0.37804898619651794, 0.3523145914077759, 0.4180276095867157, 0.3346249461174011, 0.3132311999797821, 0.40519076585769653, 0.30811065435409546, 0.2925092279911041, 0.4841303527355194, 0.38088709115982056, 0.37237435579299927, 0.480080246925354, 0.3548429012298584, 0.34750840067863464, 0.49282386898994446, 0.3548925518989563, 0.3281234800815582, 0.5909892320632935, 0.4507506191730499, 0.42628833651542664, 0.6513751149177551, 0.5311683416366577, 0.4942563772201538, 0.6639600992202759, 0.5398113131523132, 0.5092970132827759, 0.6019003391265869, 0.46907180547714233, 0.44614553451538086, 0.4839625060558319, 0.34672030806541443, 0.33124399185180664, 0.4119594693183899, 0.2771182954311371, 0.2568940818309784, 0.4870557188987732, 0.35786381363868713, 0.321870356798172, 0.5323001742362976, 0.40904730558395386, 0.364327073097229, 0.5325050354003906, 0.40779998898506165, 0.36859071254730225, 0.5039269328117371, 0.3924922049045563, 0.3535633087158203, 0.4710777699947357, 0.3841220736503601, 0.33244630694389343, 0.318568617105484, 0.2904192805290222, 0.28559598326683044, 0.09362956881523132, 0.1067713052034378, 0.12098335474729538, 0.4371216297149658, 0.3306328356266022, 0.28309813141822815, 0.46147555112838745, 0.357820063829422, 0.3089173436164856, 0.4630376994609833, 0.3586903214454651, 0.31199440360069275, 0.4513693153858185, 0.3511180281639099, 0.30648571252822876, 0.44618716835975647, 0.34593576192855835, 0.30271074175834656, 0.433458536863327, 0.32643410563468933, 0.2922746241092682, 0.40760618448257446, 0.2989181876182556, 0.2673647999763489, 0.40223103761672974, 0.3019833266735077, 0.26497700810432434, 0.42726340889930725, 0.3177323639392853, 0.28397664427757263, 0.4449828863143921, 0.3418106734752655, 0.30815035104751587, 0.4640592336654663, 0.35712307691574097, 0.3245837986469269, 0.4803270697593689, 0.37092095613479614, 0.3401428759098053, 0.4576375484466553, 0.34492722153663635, 0.31634995341300964, 0.3917165994644165, 0.2760050892829895, 0.2513902485370636, 0.3189133107662201, 0.2169780433177948, 0.1971171647310257, 0.3684769570827484, 0.25672024488449097, 0.24127213656902313, 0.5242403745651245, 0.39509692788124084, 0.3754901587963104, 0.570216953754425, 0.4295654296875, 0.40287360548973083, 0.5585350394248962, 0.4144565761089325, 0.39099881052970886, 0.5934524536132812, 0.448367178440094, 0.4259181618690491, 0.6298954486846924, 0.4891037046909332, 0.45906615257263184, 0.6326410174369812, 0.49087581038475037, 0.4571416676044464, 0.599506139755249, 0.4672522246837616, 0.4317939579486847, 0.5873756408691406, 0.4501497745513916, 0.4234960675239563, 0.40551847219467163, 0.27030321955680847, 0.2476937472820282, 0.4395073354244232, 0.3155818283557892, 0.27762261033058167, 0.5082296133041382, 0.3871646225452423, 0.34635356068611145, 0.5262638926506042, 0.4105794429779053, 0.37399837374687195, 0.5060774683952332, 0.40401801466941833, 0.3611849546432495, 0.5033100247383118, 0.3994453251361847, 0.3588030934333801, 0.2400665581226349, 0.22609317302703857, 0.24755118787288666, 0.19489014148712158, 0.20580331981182098, 0.23471783101558685, 0.42842283844947815, 0.3336835503578186, 0.2855415642261505, 0.4587797522544861, 0.3539062440395355, 0.30933016538619995, 0.46894562244415283, 0.36714401841163635, 0.3156897723674774, 0.46026110649108887, 0.35942333936691284, 0.3140644133090973, 0.4560384452342987, 0.3557523787021637, 0.3103971481323242, 0.44746896624565125, 0.3385098874568939, 0.2981940805912018, 0.4209548234939575, 0.31913214921951294, 0.28212499618530273, 0.42321810126304626, 0.32002949714660645, 0.28173181414604187, 0.42975330352783203, 0.32654640078544617, 0.29063650965690613, 0.44044938683509827, 0.3322387933731079, 0.29931625723838806, 0.44408664107322693, 0.3302520215511322, 0.29728004336357117, 0.44132861495018005, 0.32879459857940674, 0.30013778805732727, 0.4177684783935547, 0.30794718861579895, 0.27525636553764343, 0.35728615522384644, 0.24391788244247437, 0.2122863084077835, 0.3293372392654419, 0.21965816617012024, 0.19255779683589935, 0.3539752960205078, 0.2387959361076355, 0.21025116741657257, 0.4721147119998932, 0.35557377338409424, 0.3137097656726837, 0.4895021915435791, 0.3743056654930115, 0.33474788069725037, 0.4837011992931366, 0.36322447657585144, 0.31777891516685486, 0.48864614963531494, 0.35941681265830994, 0.3200148642063141, 0.5087587833404541, 0.37971070408821106, 0.3426871597766876, 0.5006211996078491, 0.3666536509990692, 0.33924537897109985, 0.4446695148944855, 0.33500686287879944, 0.2998301684856415, 0.4493674635887146, 0.3361019790172577, 0.29818347096443176, 0.39329177141189575, 0.26999959349632263, 0.23612257838249207, 0.4195915758609772, 0.3009665012359619, 0.26572170853614807, 0.4678988456726074, 0.36216387152671814, 0.32291242480278015, 0.5010265707969666, 0.3974670469760895, 0.3593275249004364, 0.502596914768219, 0.4068032205104828, 0.3641446828842163, 0.4974696636199951, 0.3975204527378082, 0.36517494916915894, 0.1881149560213089, 0.18197016417980194, 0.21724864840507507, 0.19793733954429626, 0.21777835488319397, 0.2500084638595581, 0.402727872133255, 0.3129904568195343, 0.2773401439189911, 0.44036975502967834, 0.34236225485801697, 0.2995585501194, 0.46335259079933167, 0.36701181530952454, 0.30525970458984375, 0.46654021739959717, 0.3603267967700958, 0.29791438579559326, 0.46876490116119385, 0.36285918951034546, 0.30199313163757324, 0.45942601561546326, 0.3446079194545746, 0.2995063066482544, 0.4304116666316986, 0.33052903413772583, 0.2907065749168396, 0.4290522038936615, 0.32615748047828674, 0.2872665226459503, 0.4377105236053467, 0.33481565117836, 0.29723265767097473, 0.43990251421928406, 0.3345400393009186, 0.29976215958595276, 0.4390544295310974, 0.3328055143356323, 0.29921790957450867, 0.42000719904899597, 0.3068011999130249, 0.27223479747772217, 0.38024821877479553, 0.2669428884983063, 0.23402105271816254, 0.3569372892379761, 0.24358539283275604, 0.21190296113491058, 0.38505545258522034, 0.27105259895324707, 0.2393544763326645, 0.369067519903183, 0.2580450177192688, 0.22595569491386414, 0.2993108630180359, 0.2114417403936386, 0.19033946096897125, 0.2578231990337372, 0.18959026038646698, 0.16019994020462036, 0.2419763058423996, 0.17828136682510376, 0.14548709988594055, 0.2836957275867462, 0.20853203535079956, 0.181608185172081, 0.31478720903396606, 0.22712133824825287, 0.20234695076942444, 0.3187723457813263, 0.23186564445495605, 0.20601268112659454, 0.3596891760826111, 0.2689960300922394, 0.2511470317840576, 0.4412495195865631, 0.33536845445632935, 0.30968034267425537, 0.44628119468688965, 0.3334369957447052, 0.2982431650161743, 0.4172348976135254, 0.30266809463500977, 0.27119770646095276, 0.4351745545864105, 0.32402950525283813, 0.2920377850532532, 0.4489201605319977, 0.34780094027519226, 0.3123832046985626, 0.47626662254333496, 0.3837118148803711, 0.3432295620441437, 0.4377575218677521, 0.3613847494125366, 0.3366880714893341, 0.1797974705696106, 0.1910061538219452, 0.2218654453754425, 0.2155839055776596, 0.23862327635288239, 0.2719997763633728, 0.3905743956565857, 0.3098941445350647, 0.26869264245033264, 0.4074859619140625, 0.3215115964412689, 0.27689462900161743, 0.45417171716690063, 0.3545639216899872, 0.2966912090778351, 0.46481212973594666, 0.358207106590271, 0.30953502655029297, 0.4718596637248993, 0.3652544915676117, 0.31504279375076294, 0.45433738827705383, 0.34664109349250793, 0.3036288321018219, 0.4341503381729126, 0.33158326148986816, 0.2971494495868683, 0.41065770387649536, 0.30637773871421814, 0.2701326906681061, 0.4317363500595093, 0.32745638489723206, 0.28171777725219727, 0.41307520866394043, 0.31099629402160645, 0.27620041370391846, 0.3916144371032715, 0.2905140519142151, 0.25746268033981323, 0.37927186489105225, 0.2742498517036438, 0.24287042021751404, 0.37306609749794006, 0.2713707983493805, 0.23816142976284027, 0.38098907470703125, 0.27952587604522705, 0.2492283135652542, 0.3959205448627472, 0.29647505283355713, 0.269081175327301, 0.3961905241012573, 0.3047523498535156, 0.27946844696998596, 0.34483376145362854, 0.26431381702423096, 0.24868233501911163, 0.3007031977176666, 0.23604312539100647, 0.218012273311615, 0.2806932032108307, 0.22381334006786346, 0.19868481159210205, 0.2635052502155304, 0.20306351780891418, 0.18503814935684204, 0.2817831039428711, 0.21837003529071808, 0.19796590507030487, 0.3430701792240143, 0.2682088315486908, 0.2436428815126419, 0.3916367292404175, 0.3018069267272949, 0.2889409065246582, 0.43852221965789795, 0.3419172763824463, 0.32547664642333984, 0.46114060282707214, 0.3605760931968689, 0.3306681215763092, 0.4436955153942108, 0.3360002338886261, 0.30437564849853516, 0.4222894012928009, 0.31335774064064026, 0.27944284677505493, 0.42078033089637756, 0.3240431547164917, 0.2870319187641144, 0.45193901658058167, 0.3702857494354248, 0.32770612835884094, 0.33448857069015503, 0.2908974289894104, 0.2740699052810669, 0.15879300236701965, 0.171889528632164, 0.20577119290828705, 0.19411985576152802, 0.21322645246982574, 0.24809874594211578, 0.3912407457828522, 0.3120884895324707, 0.2575046718120575, 0.36167070269584656, 0.2819289267063141, 0.22483675181865692, 0.41508427262306213, 0.33094003796577454, 0.2763257324695587, 0.4402449131011963, 0.3458718955516815, 0.2948952913284302, 0.4400727152824402, 0.345696359872818, 0.2945983409881592, 0.42707276344299316, 0.3326467275619507, 0.29128751158714294, 0.41699331998825073, 0.3194100558757782, 0.28364628553390503, 0.40900930762290955, 0.3135683536529541, 0.2741967439651489, 0.38430383801460266, 0.2888629734516144, 0.2466820478439331, 0.3531675338745117, 0.2589404881000519, 0.22428485751152039, 0.3457898199558258, 0.25159505009651184, 0.22027453780174255, 0.3691244423389435, 0.27475839853286743, 0.23971214890480042, 0.3845157325267792, 0.29021283984184265, 0.25877970457077026, 0.35601869225502014, 0.2727261185646057, 0.2414645552635193, 0.32850271463394165, 0.2597808241844177, 0.23388847708702087, 0.32795390486717224, 0.2570822238922119, 0.23662367463111877, 0.34058719873428345, 0.25843286514282227, 0.23422324657440186, 0.3340073525905609, 0.25479963421821594, 0.22522464394569397, 0.34785425662994385, 0.2734476327896118, 0.2464972585439682, 0.3390657901763916, 0.2591093182563782, 0.23581375181674957, 0.3431992530822754, 0.26525643467903137, 0.2342424988746643, 0.3734915256500244, 0.2843714952468872, 0.2571309208869934, 0.3735499680042267, 0.28594866394996643, 0.25888118147850037, 0.3832073509693146, 0.3027070462703705, 0.27725061774253845, 0.40321996808052063, 0.3280051350593567, 0.29409271478652954, 0.42102134227752686, 0.336862176656723, 0.3010765314102173, 0.4187694489955902, 0.3260921537876129, 0.2869237959384918, 0.4232824146747589, 0.33345356583595276, 0.2939968407154083, 0.4061415493488312, 0.3365482687950134, 0.30219975113868713, 0.18846502900123596, 0.18218110501766205, 0.18718835711479187, 0.14311754703521729, 0.15868337452411652, 0.17994651198387146, 0.17174986004829407, 0.19514022767543793, 0.21711868047714233, 0.37830841541290283, 0.29668521881103516, 0.2449273318052292, 0.35335955023765564, 0.26627784967422485, 0.2238943725824356, 0.30415430665016174, 0.2442532628774643, 0.215567484498024, 0.3895694613456726, 0.2984156310558319, 0.2545384168624878, 0.40304434299468994, 0.3089959919452667, 0.2620140016078949, 0.39424294233322144, 0.3029538094997406, 0.2625015676021576, 0.38472720980644226, 0.2945249080657959, 0.25531208515167236, 0.37993133068084717, 0.28373000025749207, 0.24894766509532928, 0.3616989254951477, 0.26549750566482544, 0.23202940821647644, 0.34802791476249695, 0.2535853087902069, 0.22253482043743134, 0.3624004125595093, 0.26620370149612427, 0.2356756627559662, 0.37977054715156555, 0.2880077660083771, 0.2486206591129303, 0.3505837023258209, 0.2727579176425934, 0.23905563354492188, 0.2793383300304413, 0.21620745956897736, 0.1942545622587204, 0.2517412006855011, 0.1954146772623062, 0.1685894876718521, 0.3242962658405304, 0.25211718678474426, 0.22038301825523376, 0.3796235918998718, 0.286967933177948, 0.2541153132915497, 0.4138599932193756, 0.31719788908958435, 0.2789670526981354, 0.4467041790485382, 0.34304487705230713, 0.3073532283306122, 0.4599812924861908, 0.35864710807800293, 0.3203275501728058, 0.45193809270858765, 0.3497561514377594, 0.3087954819202423, 0.44097548723220825, 0.3360736668109894, 0.2953045964241028, 0.3992425203323364, 0.30946436524391174, 0.28308621048927307, 0.3079884648323059, 0.24239996075630188, 0.21781229972839355, 0.2755514085292816, 0.22836178541183472, 0.19382135570049286, 0.3540881872177124, 0.29055291414260864, 0.24770832061767578, 0.4196144640445709, 0.33235666155815125, 0.2870313227176666, 0.4350508749485016, 0.33937665820121765, 0.29406946897506714, 0.31474769115448, 0.263139933347702, 0.24341817200183868, 0.10380683839321136, 0.10470215231180191, 0.12284862995147705, 0.1276593804359436, 0.1285875141620636, 0.1473180055618286, 0.15293943881988525, 0.1703735738992691, 0.19239521026611328, 0.35891255736351013, 0.287391722202301, 0.23440797626972198, 0.3919958472251892, 0.3022959232330322, 0.2509544789791107, 0.29605209827423096, 0.2315157949924469, 0.19278235733509064, 0.3101159632205963, 0.240363210439682, 0.21118271350860596, 0.35290852189064026, 0.2642700970172882, 0.23252199590206146, 0.3643885850906372, 0.2739732265472412, 0.23475322127342224, 0.3575502634048462, 0.2673543095588684, 0.22813859581947327, 0.35535311698913574, 0.25758326053619385, 0.22091951966285706, 0.3456811308860779, 0.2476433366537094, 0.20843419432640076, 0.34204888343811035, 0.24830403923988342, 0.21661652624607086, 0.3787807822227478, 0.2848089933395386, 0.25343775749206543, 0.34750524163246155, 0.2630572021007538, 0.23531675338745117, 0.2672284245491028, 0.20834940671920776, 0.1855371594429016, 0.21590855717658997, 0.1554258167743683, 0.14885744452476501, 0.1823762059211731, 0.1216733455657959, 0.12762236595153809, 0.20416255295276642, 0.13047800958156586, 0.13215617835521698, 0.22181648015975952, 0.1437995582818985, 0.13109785318374634, 0.2315954566001892, 0.1479463428258896, 0.13801346719264984, 0.27379167079925537, 0.1849582940340042, 0.1628599762916565, 0.3669802248477936, 0.2567112445831299, 0.2417837232351303, 0.28941190242767334, 0.18858453631401062, 0.17859552800655365, 0.23632170259952545, 0.1380232870578766, 0.1288120448589325, 0.2410704642534256, 0.15255165100097656, 0.1374586820602417, 0.25660979747772217, 0.17748497426509857, 0.15808624029159546, 0.23966532945632935, 0.18054009974002838, 0.1495990753173828, 0.30569034814834595, 0.24298040568828583, 0.19928601384162903, 0.4022636413574219, 0.31737270951271057, 0.2727881371974945, 0.4286837875843048, 0.34220847487449646, 0.30484482645988464, 0.1698978990316391, 0.1470533311367035, 0.14701992273330688, 0.08679303526878357, 0.09004709869623184, 0.10577280819416046, 0.11211475729942322, 0.11370927840471268, 0.12915125489234924, 0.1326388716697693, 0.14211425185203552, 0.16115693747997284, 0.35997340083122253, 0.299539715051651, 0.24636022746562958, 0.3994350731372833, 0.3061348497867584, 0.25334084033966064, 0.3707541227340698, 0.2844918668270111, 0.23586204648017883, 0.31037193536758423, 0.23515178263187408, 0.1968383938074112, 0.30573272705078125, 0.2298656553030014, 0.19450616836547852, 0.33101126551628113, 0.2477344572544098, 0.20594483613967896, 0.33700627088546753, 0.2480195015668869, 0.20840199291706085, 0.3460718095302582, 0.2541486322879791, 0.21493467688560486, 0.3572431206703186, 0.2643513083457947, 0.22513233125209808, 0.3700489103794098, 0.2820158898830414, 0.2443634569644928, 0.36359667778015137, 0.27260103821754456, 0.23608580231666565, 0.30265915393829346, 0.23109960556030273, 0.1945960968732834, 0.29550275206565857, 0.2324497103691101, 0.19290532171726227, 0.34649714827537537, 0.2561846375465393, 0.2286154180765152, 0.33345675468444824, 0.231301948428154, 0.2223554104566574, 0.31754711270332336, 0.21305690705776215, 0.21480748057365417, 0.331127405166626, 0.20775790512561798, 0.2252756953239441, 0.32748034596443176, 0.19845706224441528, 0.21230058372020721, 0.31561219692230225, 0.1838812232017517, 0.18776920437812805, 0.3401087820529938, 0.2072177678346634, 0.21279121935367584, 0.37313610315322876, 0.23380403220653534, 0.24153338372707367, 0.3944458067417145, 0.25051093101501465, 0.2585707902908325, 0.3989567756652832, 0.26700201630592346, 0.2537267208099365, 0.4142303764820099, 0.30052876472473145, 0.272580087184906, 0.3438029885292053, 0.25818943977355957, 0.2180141806602478, 0.29946979880332947, 0.23104612529277802, 0.18453440070152283, 0.3610772490501404, 0.28270745277404785, 0.23645152151584625, 0.24061284959316254, 0.1957537978887558, 0.18459895253181458, 0.06053026393055916, 0.057630233466625214, 0.07380522787570953, 0.06519301980733871, 0.06653773784637451, 0.08235497772693634, 0.090444415807724, 0.09090770035982132, 0.10659194737672806, 0.11605038493871689, 0.1148412898182869, 0.141224667429924, 0.3768567442893982, 0.31050899624824524, 0.26449936628341675, 0.39690130949020386, 0.3151021897792816, 0.26192906498908997, 0.37497660517692566, 0.2913639545440674, 0.23840026557445526, 0.34158822894096375, 0.26915934681892395, 0.22234900295734406, 0.2909645736217499, 0.21657882630825043, 0.1877867430448532, 0.3109205663204193, 0.2366279661655426, 0.19218988716602325, 0.32680678367614746, 0.24247756600379944, 0.20192347466945648, 0.3305278718471527, 0.24338656663894653, 0.20441432297229767, 0.3427681624889374, 0.25624725222587585, 0.21507766842842102, 0.354891836643219, 0.2655620574951172, 0.22562047839164734, 0.3348042368888855, 0.2472817748785019, 0.2080661803483963, 0.3324098289012909, 0.24036888778209686, 0.20726004242897034, 0.39464879035949707, 0.2940945029258728, 0.25587913393974304, 0.41882234811782837, 0.30672091245651245, 0.2647250294685364, 0.3946444094181061, 0.28063520789146423, 0.25244808197021484, 0.3917849063873291, 0.27855122089385986, 0.257232666015625, 0.3865707218647003, 0.2590900659561157, 0.24753499031066895, 0.4034850597381592, 0.2577851414680481, 0.26118937134742737, 0.41803935170173645, 0.263925701379776, 0.2643263339996338, 0.4289178252220154, 0.27562955021858215, 0.2768743336200714, 0.43781331181526184, 0.28052499890327454, 0.2734832167625427, 0.43844228982925415, 0.2860087752342224, 0.2698059380054474, 0.43357640504837036, 0.30605122447013855, 0.2706475257873535, 0.43440043926239014, 0.3219136893749237, 0.2809891700744629, 0.3720029592514038, 0.2833656966686249, 0.24461810290813446, 0.3295329213142395, 0.2510387599468231, 0.2025931179523468, 0.2836240828037262, 0.21889159083366394, 0.19055792689323425, 0.09441434592008591, 0.07954451441764832, 0.09323222190141678, 0.05692335590720177, 0.053195782005786896, 0.07285773009061813, 0.05775877460837364, 0.05355644226074219, 0.07307790964841843, 0.07433853298425674, 0.07206252962350845, 0.08802063018083572, 0.09528662264347076, 0.10054593533277512, 0.1194891557097435, 0.3205074667930603, 0.2661168873310089, 0.23169253766536713, 0.3847554326057434, 0.306988924741745, 0.26297977566719055, 0.35748422145843506, 0.28247758746147156, 0.24116769433021545, 0.33894044160842896, 0.267945259809494, 0.22237469255924225, 0.3058522045612335, 0.23442895710468292, 0.19370205700397491, 0.2906230092048645, 0.2151012271642685, 0.17552706599235535, 0.31213587522506714, 0.22476953268051147, 0.18470270931720734, 0.3221904933452606, 0.23286210000514984, 0.19694183766841888, 0.3388427793979645, 0.25452297925949097, 0.21419554948806763, 0.3553786873817444, 0.2686300575733185, 0.228678360581398, 0.3487916886806488, 0.2594556212425232, 0.2202376425266266, 0.3776966333389282, 0.2795056700706482, 0.2372332066297531, 0.40631330013275146, 0.3038935363292694, 0.26080694794654846, 0.4213545024394989, 0.3178301751613617, 0.27028629183769226, 0.40674442052841187, 0.30190807580947876, 0.2624236047267914, 0.37753912806510925, 0.2758729159832001, 0.23684890568256378, 0.3432554602622986, 0.2520887553691864, 0.21907787024974823, 0.31223490834236145, 0.22242173552513123, 0.20237469673156738, 0.2733328938484192, 0.20117305219173431, 0.1798417717218399, 0.2796160876750946, 0.20110665261745453, 0.17786115407943726, 0.31199023127555847, 0.21917805075645447, 0.19356834888458252, 0.3618171811103821, 0.2587790787220001, 0.22972474992275238, 0.4135868549346924, 0.31244930624961853, 0.26784375309944153, 0.4173538088798523, 0.31978362798690796, 0.2705287039279938, 0.41208961606025696, 0.3141983449459076, 0.2605503797531128, 0.374398410320282, 0.29248806834220886, 0.25592702627182007, 0.2025647759437561, 0.17628325521945953, 0.19080236554145813, 0.08877390623092651, 0.08863833546638489, 0.11429381370544434, 0.06200501695275307, 0.05951526388525963, 0.07936783879995346, 0.051934968680143356, 0.048294682055711746, 0.06753546744585037, 0.06313289701938629, 0.0608552061021328, 0.0788164883852005, 0.0789060890674591, 0.08656430244445801, 0.0983358696103096, 0.2279871106147766, 0.19783613085746765, 0.1777815967798233, 0.37354862689971924, 0.30181974172592163, 0.25270143151283264, 0.35343387722969055, 0.2805175185203552, 0.2383599430322647, 0.3339652121067047, 0.2627294063568115, 0.21687597036361694, 0.30266088247299194, 0.23203454911708832, 0.18746653199195862, 0.2715403437614441, 0.2028447538614273, 0.15872761607170105, 0.2636840045452118, 0.1925869584083557, 0.16246820986270905, 0.29668521881103516, 0.2215576171875, 0.18224522471427917, 0.33931031823158264, 0.25837695598602295, 0.20876741409301758, 0.3786585032939911, 0.29321205615997314, 0.2481280267238617, 0.402462899684906, 0.3117884397506714, 0.2715264856815338, 0.42489439249038696, 0.32412150502204895, 0.2795164883136749, 0.414354145526886, 0.3096831738948822, 0.26191431283950806, 0.4153645932674408, 0.3120562732219696, 0.26732176542282104, 0.4176509380340576, 0.3173699378967285, 0.2733042240142822, 0.40017926692962646, 0.30279165506362915, 0.25801724195480347, 0.3863217234611511, 0.29234179854393005, 0.25043758749961853, 0.36781394481658936, 0.2755894958972931, 0.23546800017356873, 0.3402005434036255, 0.254433274269104, 0.2143813818693161, 0.3431774973869324, 0.2528739273548126, 0.2169681042432785, 0.3788807988166809, 0.2860671877861023, 0.24686092138290405, 0.4311177730560303, 0.32854244112968445, 0.2829004228115082, 0.46707582473754883, 0.36087921261787415, 0.314941942691803, 0.4643939435482025, 0.3647955656051636, 0.3161005973815918, 0.43162715435028076, 0.34835419058799744, 0.3007374405860901, 0.29567039012908936, 0.25289666652679443, 0.2570957839488983, 0.161971777677536, 0.15881678462028503, 0.1940806359052658, 0.11398768424987793, 0.11007076501846313, 0.1372816264629364, 0.07073362171649933, 0.06681545823812485, 0.08579320460557938, 0.05760563910007477, 0.05135631561279297, 0.06961806863546371, 0.055793773382902145, 0.05233501270413399, 0.06979355216026306, 0.06256303936243057, 0.0686858519911766, 0.08493249863386154, 0.1518678367137909, 0.14091035723686218, 0.12833893299102783, 0.3470573425292969, 0.28164246678352356, 0.23568882048130035, 0.3554447889328003, 0.2880769968032837, 0.24101920425891876, 0.3309482932090759, 0.26055774092674255, 0.21349923312664032, 0.3060360550880432, 0.23545947670936584, 0.18840138614177704, 0.2721025049686432, 0.2063666731119156, 0.16597633063793182, 0.22256238758563995, 0.16965965926647186, 0.14129148423671722, 0.20511357486248016, 0.1581941843032837, 0.13222046196460724, 0.25196951627731323, 0.20345176756381989, 0.16904029250144958, 0.33347630500793457, 0.26294490694999695, 0.2253090888261795, 0.390899121761322, 0.3050629198551178, 0.26085367798805237, 0.44359955191612244, 0.3420328199863434, 0.30194157361984253, 0.45829710364341736, 0.35765570402145386, 0.3122677803039551, 0.452980101108551, 0.3509868085384369, 0.30070605874061584, 0.4659992754459381, 0.36024004220962524, 0.3074303865432739, 0.4695497751235962, 0.3640454411506653, 0.32167625427246094, 0.46484827995300293, 0.35496097803115845, 0.31577473878860474, 0.46057796478271484, 0.3450596034526825, 0.3114539384841919, 0.44464439153671265, 0.33276236057281494, 0.30804651975631714, 0.4370194375514984, 0.3373217284679413, 0.31436944007873535, 0.4653356969356537, 0.3638789653778076, 0.32108309864997864, 0.498119592666626, 0.38944295048713684, 0.3423115015029907, 0.5125730633735657, 0.40191611647605896, 0.357479453086853, 0.49191296100616455, 0.3892665505409241, 0.3442680239677429, 0.32363688945770264, 0.2907745838165283, 0.2797646224498749, 0.19840487837791443, 0.19296786189079285, 0.23477290570735931, 0.16575752198696136, 0.16432031989097595, 0.20264983177185059, 0.11678332090377808, 0.11297900974750519, 0.14717033505439758, 0.08064288645982742, 0.07860619574785233, 0.09992792457342148, 0.05956072732806206, 0.052428942173719406, 0.07487373054027557, 0.05490846186876297, 0.05086646229028702, 0.07035218179225922, 0.05796033889055252, 0.05776239186525345, 0.07700338959693909, 0.11803355813026428, 0.113003209233284, 0.11648242175579071, 0.2621428370475769, 0.22889931499958038, 0.19070293009281158, 0.36023643612861633, 0.29658418893814087, 0.24453267455101013, 0.34532666206359863, 0.2738797664642334, 0.226822629570961, 0.32393530011177063, 0.25334399938583374, 0.206288680434227, 0.2839506268501282, 0.2204667180776596, 0.18433444201946259, 0.2512064576148987, 0.19569158554077148, 0.16428060829639435, 0.22924548387527466, 0.17850469052791595, 0.1445036083459854, 0.20527882874011993, 0.1614913046360016, 0.13028858602046967, 0.21464945375919342, 0.16960494220256805, 0.14080792665481567, 0.28392350673675537, 0.22737345099449158, 0.1880790740251541, 0.35884955525398254, 0.2952265739440918, 0.24995753169059753, 0.4204166531562805, 0.348469614982605, 0.2935921847820282, 0.45606231689453125, 0.3666648268699646, 0.31178027391433716, 0.4690937101840973, 0.3648296296596527, 0.3183821439743042, 0.47141847014427185, 0.3718341588973999, 0.3321603834629059, 0.45576146245002747, 0.36486098170280457, 0.32564207911491394, 0.4470905661582947, 0.3545631766319275, 0.3141447603702545, 0.42806220054626465, 0.3353632092475891, 0.29336240887641907, 0.4161829352378845, 0.32281094789505005, 0.29451408982276917, 0.44480082392692566, 0.3580997586250305, 0.32666707038879395, 0.4683583080768585, 0.3810123801231384, 0.33786633610725403, 0.46681585907936096, 0.3799036145210266, 0.3259696960449219, 0.37381163239479065, 0.3128948509693146, 0.282091349363327, 0.18427953124046326, 0.17927372455596924, 0.19951780140399933, 0.20073753595352173, 0.20105597376823425, 0.23689129948616028, 0.16982242465019226, 0.17175526916980743, 0.20913481712341309, 0.12101823091506958, 0.12106887251138687, 0.15606340765953064, 0.09349783509969711, 0.09349797666072845, 0.11543530970811844, 0.067892886698246, 0.06247739493846893, 0.08358284085988998, 0.059332702308893204, 0.05404670909047127, 0.07757234573364258, 0.05445333570241928, 0.05053466930985451, 0.07405777275562286, 0.11782607436180115, 0.11677709221839905, 0.13158656656742096, 0.19788651168346405, 0.18112941086292267, 0.15197035670280457, 0.3223198652267456, 0.26301366090774536, 0.21313992142677307, 0.34414422512054443, 0.27564680576324463, 0.23104754090309143, 0.3294902443885803, 0.2549937665462494, 0.21198365092277527, 0.2886069416999817, 0.22361278533935547, 0.18753395974636078, 0.2647063434123993, 0.20516754686832428, 0.17036466300487518, 0.2563076913356781, 0.19477002322673798, 0.16730241477489471, 0.2478274255990982, 0.19446679949760437, 0.15884196758270264, 0.23743779957294464, 0.18244530260562897, 0.15615808963775635, 0.20394690334796906, 0.15355117619037628, 0.12911809980869293, 0.23583731055259705, 0.18852350115776062, 0.15725922584533691, 0.29611894488334656, 0.24500098824501038, 0.20821718871593475, 0.3582163453102112, 0.29663440585136414, 0.2579076588153839, 0.385303258895874, 0.31157293915748596, 0.271257221698761, 0.3796081840991974, 0.3029095530509949, 0.2629391849040985, 0.3863970637321472, 0.3133694529533386, 0.26979923248291016, 0.3867761194705963, 0.31672191619873047, 0.2726743519306183, 0.39957696199417114, 0.32594284415245056, 0.30238085985183716, 0.37554457783699036, 0.30570974946022034, 0.28820720314979553, 0.3537854254245758, 0.2916657030582428, 0.2684849202632904, 0.34340745210647583, 0.2884594202041626, 0.25265753269195557, 0.3077754080295563, 0.26390525698661804, 0.23513194918632507, 0.14035974442958832, 0.12262352555990219, 0.13427278399467468, 0.1513618528842926, 0.1502051055431366, 0.18397819995880127, 0.20885345339775085, 0.2054215967655182, 0.24475358426570892, 0.17767435312271118, 0.17754913866519928, 0.21696382761001587, 0.14005038142204285, 0.13625413179397583, 0.17522843182086945, 0.10277199000120163, 0.10448146611452103, 0.128447026014328, 0.0765993744134903, 0.07362843304872513, 0.09668435156345367, 0.06396423280239105, 0.060216255486011505, 0.08357217907905579, 0.05891770124435425, 0.05505259335041046, 0.07546728849411011, 0.14682358503341675, 0.14134719967842102, 0.17187921702861786, 0.1468362957239151, 0.13885992765426636, 0.1332782357931137, 0.24624399840831757, 0.20596030354499817, 0.18278400599956512, 0.2951413094997406, 0.24107296764850616, 0.2135053426027298, 0.30359920859336853, 0.23483648896217346, 0.20272596180438995, 0.29150885343551636, 0.2287619262933731, 0.19485360383987427, 0.2731471359729767, 0.20844461023807526, 0.17609591782093048, 0.25000542402267456, 0.18885381519794464, 0.16372562944889069, 0.24155227839946747, 0.18040695786476135, 0.1552727222442627, 0.23396456241607666, 0.17575620114803314, 0.14806097745895386, 0.21973104774951935, 0.16482993960380554, 0.13535557687282562, 0.21413666009902954, 0.15857747197151184, 0.13491979241371155, 0.2215985655784607, 0.16679677367210388, 0.1411820352077484, 0.2281455099582672, 0.17797398567199707, 0.15021955966949463, 0.2390497922897339, 0.19943338632583618, 0.17059139907360077, 0.2529439628124237, 0.21341125667095184, 0.18622927367687225, 0.26314494013786316, 0.21556411683559418, 0.19348149001598358, 0.26161515712738037, 0.21271145343780518, 0.1906307339668274, 0.26713693141937256, 0.2183460295200348, 0.19503825902938843, 0.25482451915740967, 0.217301607131958, 0.1948656290769577, 0.2110089510679245, 0.1802290678024292, 0.15896637737751007, 0.19710852205753326, 0.17225462198257446, 0.14946241676807404, 0.1195099949836731, 0.10307634621858597, 0.11100146174430847, 0.08763688057661057, 0.07918189465999603, 0.09731493890285492, 0.17056666314601898, 0.16775104403495789, 0.19100825488567352, 0.20822125673294067, 0.20295730233192444, 0.24412375688552856, 0.18675296008586884, 0.182834193110466, 0.2282910794019699, 0.14143426716327667, 0.13526257872581482, 0.18040606379508972, 0.1159377247095108, 0.11494612693786621, 0.1436365395784378, 0.08651692420244217, 0.09028361737728119, 0.10793907195329666, 0.06782922148704529, 0.06990204751491547, 0.08896415680646896, 0.062394995242357254, 0.05968319997191429, 0.0789392814040184, 0.15788201987743378, 0.1505262702703476, 0.19427251815795898, 0.13400991261005402, 0.1326691061258316, 0.14743638038635254, 0.17313146591186523, 0.1598084568977356, 0.14747951924800873, 0.24284791946411133, 0.202530637383461, 0.18409940600395203, 0.27101612091064453, 0.21958310902118683, 0.1967310756444931, 0.271440327167511, 0.20968249440193176, 0.1828431636095047, 0.26533734798431396, 0.2032051682472229, 0.17202691733837128, 0.2507058084011078, 0.18968036770820618, 0.16223442554473877, 0.24595269560813904, 0.18537040054798126, 0.15792541205883026, 0.24271513521671295, 0.18431150913238525, 0.15973946452140808, 0.2307634800672531, 0.17266857624053955, 0.14583130180835724, 0.22057653963565826, 0.16177250444889069, 0.13838453590869904, 0.21365539729595184, 0.15671250224113464, 0.13742132484912872, 0.19786673784255981, 0.15188975632190704, 0.12684519588947296, 0.17826762795448303, 0.1421528458595276, 0.12263613194227219, 0.1813458800315857, 0.1404218226671219, 0.12998279929161072, 0.18957117199897766, 0.14935514330863953, 0.12497827410697937, 0.19303429126739502, 0.15370726585388184, 0.13060128688812256, 0.20082621276378632, 0.15744175016880035, 0.13622519373893738, 0.20121929049491882, 0.1673770397901535, 0.14261554181575775, 0.2032840996980667, 0.15607179701328278, 0.14588850736618042, 0.1520756036043167, 0.1253795176744461, 0.11447148770093918, 0.07871030271053314, 0.07582547515630722, 0.07358794659376144, 0.09689854085445404, 0.09578366577625275, 0.11619529128074646, 0.1839647889137268, 0.18241529166698456, 0.20934723317623138, 0.20923267304897308, 0.20531117916107178, 0.2406086027622223, 0.1848069280385971, 0.18088535964488983, 0.22218650579452515, 0.15553629398345947, 0.15160788595676422, 0.19845972955226898, 0.12349515408277512, 0.12083561718463898, 0.15634198486804962, 0.09332363307476044, 0.09564301371574402, 0.11789045482873917, 0.07436249405145645, 0.07911253720521927, 0.09675682336091995, 0.06381348520517349, 0.06541824340820312, 0.0809762105345726, 0.1693267822265625, 0.16516059637069702, 0.20683442056179047, 0.15578940510749817, 0.15260492265224457, 0.17774973809719086, 0.13606496155261993, 0.12717173993587494, 0.1465865969657898, 0.18823303282260895, 0.1629595011472702, 0.14928163588047028, 0.22888168692588806, 0.18018114566802979, 0.16322442889213562, 0.2386636584997177, 0.18945500254631042, 0.16592365503311157, 0.2498445063829422, 0.19189749658107758, 0.16318580508232117, 0.24674928188323975, 0.19104671478271484, 0.16157573461532593, 0.24876636266708374, 0.19386091828346252, 0.1664099097251892, 0.25036293268203735, 0.19136649370193481, 0.16409049928188324, 0.24340543150901794, 0.18180498480796814, 0.15713179111480713, 0.23559868335723877, 0.17689763009548187, 0.15343765914440155, 0.22640734910964966, 0.1709844022989273, 0.14783784747123718, 0.216696098446846, 0.16169556975364685, 0.14001508057117462, 0.22004155814647675, 0.16736823320388794, 0.1421395093202591, 0.22163709998130798, 0.1680949628353119, 0.1428714543581009, 0.21822988986968994, 0.1725044697523117, 0.14030064642429352, 0.22202134132385254, 0.17496713995933533, 0.1437421292066574, 0.2278023660182953, 0.1797410249710083, 0.14937929809093475, 0.22271069884300232, 0.1771528124809265, 0.14578446745872498, 0.1923048347234726, 0.14698290824890137, 0.1297357827425003, 0.1279049664735794, 0.10897517949342728, 0.10275544226169586, 0.07765598595142365, 0.07294927537441254, 0.07734427601099014, 0.11293364316225052, 0.10981061309576035, 0.13082008063793182, 0.18542800843715668, 0.181507870554924, 0.21176908910274506, 0.21085672080516815, 0.20693515241146088, 0.2449825406074524, 0.1878722757101059, 0.18395066261291504, 0.2289964109659195, 0.1661536991596222, 0.15942169725894928, 0.20648036897182465, 0.13139134645462036, 0.125457763671875, 0.17067237198352814, 0.10533490777015686, 0.10131348669528961, 0.13378272950649261, 0.07523111999034882, 0.08626368641853333, 0.10060422867536545, 0.06336943805217743, 0.06900458037853241, 0.08602891117334366]])
```

**Acceptance criteria:**

1. The dropout technique is demonstrated.

**Test case:**

Due to randomness when initializing the weights, your output may be different.

```python
python task11.py
```

```console
tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.7503, 0.0000, 0.0000, 0.4613, 0.0000, 0.0000]],
       grad_fn=<MulBackward0>)
```

## Task 12

**Description:**

Randomly sample `10` values of the learning rate and momentum from the uniform distribution. Learning rates have to be between `0.01` and `0.0001`. The momentums have to be between `0.85` and `0.99`.

**Acceptance criteria:**

1. An array of tuples to use for learning rate and momentum is outputted.
2. The first element from every tuple is to be used for the learning rate.
3. The second element from every tuple is to be used for the momentum.

**Test case:**

Due to randomness when sampling, your output may be different.

```python
python task12.py
```

```console
[(0.0001366142212846236, 0.9167832165897652), (0.00020071926415008681, 0.9305219000746696), (0.00019434803670958416, 0.9754606278952971), (0.0005823673916463207, 0.9562884005232178), (0.00020528672764024125, 0.9160138637178254), (0.0043510092612040994, 0.9893088011835995), (0.0003012126498614589, 0.956004796685816), (0.0008225887209191026, 0.9544874421412471), (0.0031029995649895347, 0.8939537139053088), (0.005003994717738704, 0.981138332509216)]
```
