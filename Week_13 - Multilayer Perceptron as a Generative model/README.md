# 🎯 Goals for week 13

1. Implement a **multilayer perceptron** that can accept multiple characters and output a probability distribution for the next character in the sequence.
2. Dive into some of the internals of MLPs and scrutinize the statistics of the forward pass activations, backward pass gradients, and some of the pitfalls when they are improperly scaled.
3. Practice writing high quality code:
   1. Easy to read.
   2. Safe from bugs.
   3. Ready for change.

## Task 1 - Set up

**Description:**

Read in all words in the `names.txt` file. Output the first $8$, the total number of words and the mapping from integers.

**Acceptance criteria:**

1. The test case passes.

**Test case:**

```console
python task01.py
```

```console
First 8 words: ['emma', 'olivia', 'ava', 'isabella', 'sophia', 'charlotte', 'mia', 'amelia']
Total number of words: 32033
Integer to character map: {0: '.', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}
```

## Task 2 - Building the $X$ and $y$ datasets

**Description:**

Store the inputs of the neural network in a matrix of indices - $X$. Store their corresponding outputs in an array $y$. Use a context length of $3$.

Output their shape and data type, their entries for the first $5$ words along with their character equivalent.

**Acceptance criteria:**

1. The test case passes.
2. $X$ and $y$ are stored as `pytorch` tensors.

**Test case:**

```console
python task02.py
```

```console
emma
... -> e
..e -> m
.em -> m
emm -> a
mma -> .
olivia
... -> o
..o -> l
.ol -> i
oli -> v
liv -> i
ivi -> a
via -> .
ava
... -> a
..a -> v
.av -> a
ava -> .
isabella
... -> i
..i -> s
.is -> a
isa -> b
sab -> e
abe -> l
bel -> l
ell -> a
lla -> .
sophia
... -> s
..s -> o
.so -> p
sop -> h
oph -> i
phi -> a
hia -> .

tensor([[ 0,  0,  0],
        [ 0,  0,  5],
        [ 0,  5, 13],
        [ 5, 13, 13],
        [13, 13,  1],
        [ 0,  0,  0],
        [ 0,  0, 15],
        [ 0, 15, 12],
        [15, 12,  9],
        [12,  9, 22],
        [ 9, 22,  9],
        [22,  9,  1],
        [ 0,  0,  0],
        [ 0,  0,  1],
        [ 0,  1, 22],
        [ 1, 22,  1],
        [ 0,  0,  0],
        [ 0,  0,  9],
        [ 0,  9, 19],
        [ 9, 19,  1],
        [19,  1,  2],
        [ 1,  2,  5],
        [ 2,  5, 12],
        [ 5, 12, 12],
        [12, 12,  1],
        [ 0,  0,  0],
        [ 0,  0, 19],
        [ 0, 19, 15],
        [19, 15, 16],
        [15, 16,  8],
        [16,  8,  9],
        [ 8,  9,  1]])
tensor([ 5, 13, 13,  1,  0, 15, 12,  9, 22,  9,  1,  0,  1, 22,  1,  0,  9, 19,
         1,  2,  5, 12, 12,  1,  0, 19, 15, 16,  8,  9,  1,  0])

torch.Size([32, 3]) torch.int64 torch.Size([32]) torch.int64
```

## Task 3 - Building the embedding matrix $C$

**Description:**

We now have the examples and the labels for the first $5$ words in $X$ and $y$.

This task is about creating the embedding matrix $C$ that will map a character to a $2$-dimensional vector of numbers. In our case, the mapping is actually from the elements in the character to integer map (i.e. integers, not characters).

Create the matrix $C$, print its shape and embed the first $5$ training examples in $X$. Output the first $2$ entries in the embedded $X$.

**Acceptance criteria:**

1. The test case passes.
2. No built-in classes are used.
3. $C$ is initialized using values drawn from the normal distribution.

**Test case:**

```console
python task03.py
```

```console
Shape of C: torch.Size([27, 2])
Shape of embedded X: torch.Size([32, 3, 2])
tensor([[[ 0.0867, -0.1426],
         [ 0.0867, -0.1426],
         [ 0.0867, -0.1426]],

        [[ 0.0867, -0.1426],
         [ 0.0867, -0.1426],
         [ 1.2946,  0.3052]]])
```

## Task 4 - Building the hidden layer

**Description:**

Create the hidden layer - $W1$, that would take the embedded inputs and pass them through $100$ neurons with biases. Intentionally, do **not add** an activation function.

**Acceptance criteria:**

1. The test case passes.
2. No built-in classes are used.
3. $W1$ is initialized using values drawn from the normal distribution.

**Test case:**

```console
python task04.py
```

```console
Shape of hidden layer: torch.Size([32, 100])
```

## Task 5 - Building the output layer

**Description:**

Create the output layer - $W2$, that would take the output of the previous layer and transform it into probabilities of the next possible character.

Output the probabilities assigned to the labels of the examples.

**Acceptance criteria:**

1. The test case passes.
2. No built-in classes are used.
3. $W2$ is initialized using values drawn from the normal distribution.

**Test case:**

```console
python task05.py
```

```console
tensor([1.3670e-08, 1.4119e-05, 3.4324e-31, 2.1639e-41, 2.4317e-21, 3.3330e-09,
        5.4074e-22, 3.7765e-42, 7.9321e-18, 1.0000e+00, 0.0000e+00, 6.9925e-43,
        1.1420e-22, 1.1023e-28, 1.4370e-21, 0.0000e+00, 1.9340e-15, 4.6705e-18,
        9.6211e-26, 9.0605e-01, 4.1909e-15, 4.5216e-40, 0.0000e+00, 7.1567e-41,
        5.6870e-20, 3.1342e-12, 7.9014e-14, 9.5597e-42, 4.1560e-21, 2.3966e-04,
        1.7248e-11, 9.0756e-22])
```

## Task 6 - Adding loss

**Description:**

Let's refactor the code a little bit and output the loss.

Add a generator object with a seed of `2147483647` for reproducibility, add the `tanh` non-linearity in the hidden layer, output the number of parameters of the network and calculate the negative log likelihood using the built-in function.

**Acceptance criteria:**

1. The test case passes.

**Test case:**

```console
python task06.py
```

```console
Number of parameters: 3481
NULL: 17.769712448120117
```

## Task 7 - Training

**Description:**

We have the forward pass ready. Let's add the backward pass, the weight update logic and train for `10` epochs on the five embedded words with a learning rate of `0.1`. Output the loss after every epoch.

**Acceptance criteria:**

1. The test case passes.

**Test case:**

```console
python task07.py
```

```console
17.769712448120117
13.656402587890625
11.298768997192383
9.452457427978516
7.984262466430664
6.891321182250977
6.1000142097473145
5.452036380767822
4.8981523513793945
4.414663791656494
```

## Task 8 - Training on the full dataset

**Description:**

Train on the full dataset for `10` epochs. Output the shapes of the feature and target arrays. Output the loss before doing the backward pass and the time it took for doing the forward pass.

**Acceptance criteria:**

1. The test case passes.

**Test case:**

```console
python task08.py
```

```console
X.shape=torch.Size([228146, 3])
X.dtype=torch.int64
y.shape=torch.Size([228146])
y.dtype=torch.int64

Starting to train:

19.505229949951172, 0.2582 seconds
17.084484100341797, 0.189 seconds
15.776530265808105, 0.1876 seconds
14.833340644836426, 0.1761 seconds
14.002605438232422, 0.1872 seconds
13.253263473510742, 0.1786 seconds
12.57991886138916, 0.1791 seconds
11.983102798461914, 0.1807 seconds
11.47049331665039, 0.1886 seconds
11.05185604095459, 0.1792 seconds
```

## Task 9 - Minibatches

**Description:**

Train on the full dataset using minibatches of `32` randomly drawn examples. Output the value of the loss and the time it takes to do the forward pass for $100$ epochs.

**Acceptance criteria:**

1. The test case passes.

**Test case:**

```console
python task09.py
```

```console
19.53308868408203, 0.0018 seconds
17.153175354003906, 0.0004 seconds
16.010910034179688, 0.0003 seconds
12.494965553283691, 0.0002 seconds
12.856255531311035, 0.0002 seconds
13.110164642333984, 0.0002 seconds
10.452533721923828, 0.0003 seconds
13.025611877441406, 0.0002 seconds
13.183573722839355, 0.0002 seconds
11.7338285446167, 0.0002 seconds
12.034649848937988, 0.0002 seconds
12.248814582824707, 0.0002 seconds
10.390960693359375, 0.0002 seconds
12.706422805786133, 0.0002 seconds
8.389993667602539, 0.0002 seconds
9.72773265838623, 0.0002 seconds
8.927046775817871, 0.0003 seconds
9.600017547607422, 0.0002 seconds
9.020207405090332, 0.0002 seconds
7.288248538970947, 0.0002 seconds
8.40522575378418, 0.0002 seconds
7.6674418449401855, 0.0002 seconds
9.615256309509277, 0.0002 seconds
7.679961681365967, 0.0002 seconds
8.066514015197754, 0.0002 seconds
5.941122055053711, 0.0003 seconds
8.620623588562012, 0.0002 seconds
7.215698719024658, 0.0002 seconds
5.303076267242432, 0.0002 seconds
6.400191307067871, 0.0002 seconds
9.275991439819336, 0.0003 seconds
7.926577091217041, 0.0002 seconds
6.125301837921143, 0.0002 seconds
7.471330642700195, 0.0002 seconds
5.129348278045654, 0.0002 seconds
7.691327095031738, 0.0002 seconds
6.324587345123291, 0.0002 seconds
6.583160400390625, 0.0002 seconds
5.485723972320557, 0.0002 seconds
5.539238452911377, 0.0002 seconds
5.0138750076293945, 0.0002 seconds
7.254990577697754, 0.0002 seconds
5.602362155914307, 0.0002 seconds
6.012321472167969, 0.0002 seconds
5.3638997077941895, 0.0002 seconds
5.8396220207214355, 0.0002 seconds
4.563700199127197, 0.0002 seconds
6.45138692855835, 0.0002 seconds
5.933651447296143, 0.0003 seconds
6.572027206420898, 0.0002 seconds
5.648553848266602, 0.0002 seconds
5.038222789764404, 0.0002 seconds
7.378749370574951, 0.0002 seconds
5.064876556396484, 0.0002 seconds
7.604363441467285, 0.0002 seconds
4.865014553070068, 0.0002 seconds
5.959437370300293, 0.0002 seconds
5.355529308319092, 0.0002 seconds
4.112236499786377, 0.0002 seconds
5.137135982513428, 0.0002 seconds
3.9323277473449707, 0.0002 seconds
4.6108293533325195, 0.0002 seconds
5.1367926597595215, 0.0003 seconds
4.79396915435791, 0.0002 seconds
5.785887241363525, 0.0002 seconds
3.999778985977173, 0.0002 seconds
4.602226734161377, 0.0002 seconds
4.6396684646606445, 0.0002 seconds
3.9496984481811523, 0.0002 seconds
4.239940166473389, 0.0002 seconds
5.5005364418029785, 0.0002 seconds
4.533195972442627, 0.0002 seconds
3.9535837173461914, 0.0002 seconds
4.273637294769287, 0.0003 seconds
3.216712236404419, 0.0002 seconds
4.331841945648193, 0.0003 seconds
5.60276460647583, 0.0002 seconds
4.381503582000732, 0.0002 seconds
4.042550563812256, 0.0002 seconds
3.9339659214019775, 0.0002 seconds
4.122404098510742, 0.0002 seconds
3.980462074279785, 0.0002 seconds
3.21096134185791, 0.0002 seconds
5.3693342208862305, 0.0002 seconds
4.211599826812744, 0.0002 seconds
3.8313796520233154, 0.0003 seconds
4.7066192626953125, 0.0002 seconds
3.732875347137451, 0.0002 seconds
3.550443172454834, 0.0002 seconds
4.344723224639893, 0.0002 seconds
3.6195812225341797, 0.0002 seconds
4.272087574005127, 0.0002 seconds
4.700298309326172, 0.0002 seconds
3.762676239013672, 0.0002 seconds
3.203040599822998, 0.0002 seconds
4.566572189331055, 0.0002 seconds
3.56540846824646, 0.0002 seconds
4.293333053588867, 0.0002 seconds
2.7590601444244385, 0.0002 seconds
2.990215301513672, 0.0003 seconds
```

## Task 10 - Optimizing the learning rate

**Description:**

Choose the best value for the learning rate, if training for `1,000` epochs. Perform loss analysis for different values, output the loss vs learning rate plot and write in a comment the value that you think is most appropriate.

**Acceptance criteria:**

1. The test case passes.
2. A comment is written with the optimal value for the learning rate.

**Test case:**

```console
python task10.py
```

The following plot is created:

![w13_lr_plot.png](../assets/w13_lr_plot.png?raw=true "w13_lr_plot.png")

## Task 11 - Learning rate decay

**Description:**

Train for `50,000` epochs with a learning rate of `0.1`. Then decay (i.e. lower) the learning rate to `0.01` and train for another `10,000` epochs. Output the obtained loss on the full training set.

**Acceptance criteria:**

1. The test case passes.

**Test case:**

```console
python task11.py
```

```console
Decayed learning rate to 0.01
Loss: 2.2973411083221436
```

## Task 12 - Data splits

**Description:**

Split the labelled data into 80% `train`, 10% `dev` and 10% `test`. Output the number of examples from each feature and target tensors. Research how you can use [torch.tensor_split](https://pytorch.org/docs/stable/generated/torch.tensor_split.html#torch-tensor-split).

Train for `50,000` epochs with a learning rate of `0.1`. Then, for the final `10,000`, decay it to `0.01`. Output the obtained loss on the training and developments sets.

**Acceptance criteria:**

1. The test case passes.

**Test case:**

```console
python task12.py
```

```console
torch.Size([182516, 3]) torch.Size([182516])
torch.Size([22815, 3]) torch.Size([22815])
torch.Size([22815, 3]) torch.Size([22815])
Training loss: 2.249359130859375
Validation loss: 2.48797607421875
```

## Task 13 - Bigger Neural Net

**Description:**

Let's increase the size of our neural network. Set the number of neurons in the hidden layer to `300`. Output the number of parameters.

Train for `90,000` epochs with a learning rate of `0.1`. Then, for the final `10,000`, decay it to `0.01`. Output the obtained loss on the training and developments sets. Plot the two losses against the steps.

**Acceptance criteria:**

1. The test case passes.

**Test case:**

```console
python task13.py
```

```console
Number of parameters: 10281
Decaying learning rate at the 90000 epoch.
Training loss: 2.189983367919922
Validation loss: 2.452975034713745
```

and the following plot is created:

![w13_step_vs_loss.png](../assets/w13_step_vs_loss.png?raw=true "w13_step_vs_loss.png")

## Task 14 - What did the neural net learn?

**Description:**

Let's visualize the embeddings of the characters after training the network to see which characters are placed together and which are far apart.

**Acceptance criteria:**

1. The test case passes.

**Test case:**

```console
python task14.py
```

The following plot is created:

![w13_embeddings.png](../assets/w13_embeddings.png?raw=true "w13_embeddings.png")

## Task 15 - Increasing the embedding size

**Description:**

Embed characters in `10`-dimensional space and decrease the number of neurons in the hidden layer to `200`. Also, because when plotting the pure loss, we get a "hockey-stick" graph, let's try plotting the logarithm to the base `10` of the loss.

Train for `100,000` epochs with a learning rate of `0.1` and then, for the next `100,000`, decay it to `0.01`.

**Acceptance criteria:**

1. The test case passes.

**Test case:**

```console
python task15.py
```

```console
Number of parameters: 11897
Decaying learning rate at the 50000 epoch.
Training loss: 2.112290859222412
Validation loss: 2.4228179454803467
```

![w13_larger_embeddings.png](../assets/w13_larger_embeddings.png?raw=true "w13_larger_embeddings.png")

## Task 16 - Improving the model

**Description:**

Try to improve the model. Do at least $5$ experiments, tweaking various parameters, in an attempt to decrease the validation loss even more.

**Acceptance criteria:**

1. A model that achieves a lower validation loss.

## Task 17 - Sampling from the model

**Description:**

Use the best setup you've created so far (the one that achieves the lowest development loss) and sample `20` words from the model. Reinitialize the generator object with a seed of `2147483657`.

**Acceptance criteria:**

1. The test case passes.

**Test case:**

```console
python task17.py
```

```console
mona.
kayah.
see.
med.
rylla.
emmaniendra.
gracee.
daeliah.
miloelle.
elieanana.
leigh.
malaia.
noshuder.
shiriel.
kinde.
jennox.
teus.
kunzey.
dariyah.
faeh.
```

## Task 18 - Refactoring

**Description:**

Take some time to refactor your code and move parts of it into functions. In particular:

- Define a `build_dataset` function that takes in the words, the block size and the mapping between characters and integers, and outputs the tensors $X$ and $y$;
- If you haven't done so already, when constructing the minibatch, use the generator object as an argument to `torch.randint`;
- Output the loss every `10_000` iterations.

Make sure your output matches the structure shown in the test case below as close as possible.

**Acceptance criteria:**

1. The structure of the output matches the one in the test case.

**Test case:**

If we have the following setup:

- a generator object with `seed=2147483657`;
- a learning rate of `0.1`;
- batch size of `32`;
- block size of $3$;
- training for `200_000` epochs;
- and we do sampling right after training.

then running this:

```console
python task18.py
```

should produce this:

```console
Number of words: 32033
itos={0: '.', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}
vocab_size=27
Training dataset shape:
torch.Size([182625, 3]) torch.Size([182625])
Validation dataset shape:
torch.Size([22655, 3]) torch.Size([22655])
Testing dataset shape:
torch.Size([22866, 3]) torch.Size([22866])
Number of parameters: 11897
      0/ 200000: 27.8817
  10000/ 200000: 2.8341
  20000/ 200000: 2.5523
  30000/ 200000: 2.8717
  40000/ 200000: 2.0855
  50000/ 200000: 2.5842
  60000/ 200000: 2.4150
  70000/ 200000: 2.1321
  80000/ 200000: 2.3674
  90000/ 200000: 2.3077
Decaying learning rate at the 100000 epoch.
 100000/ 200000: 2.0464
 110000/ 200000: 2.4816
 120000/ 200000: 1.9383
 130000/ 200000: 2.4820
 140000/ 200000: 2.1662
 150000/ 200000: 2.1765
 160000/ 200000: 2.0685
 170000/ 200000: 1.7901
 180000/ 200000: 2.0546
 190000/ 200000: 1.8380
train 2.125401020050049
val 2.1713311672210693
mora.
mayah.
seel.
nah.
yam.
rensleighdrae.
caileed.
elin.
shy.
jen.
eden.
estanaraelyn.
malke.
cayshuberlyni.
jest.
jair.
jenipanthono.
ubelleda.
kylynn.
els.
```

## Task 19

**Description:**

We ought to investigate the initial high value for the loss. We suspect that the logits might very different from one another. Output the first row of logits that come out from the very first iteration. They should be:

- very high or very low numbers;
- the difference between each of them is large (they are not very close to one another).

Then, apply the suggestions in `notes.md`, let the network train for the full amount of epochs and plot the recorded loss to see if that "hockey stick" appearance is still there.

Output also the training and validation losses and compare the results with the ones seen previously.

**Acceptance criteria:**

1. The test case passes.
2. An analysis of the results is performed.

**Test case:**

```console
python task19.py
```

```console
      0/ 200000: 27.8817
logits = tensor([ -2.3527,  36.4366, -10.7306,   5.7165,  18.6409, -11.6998,  -2.1991,
          1.8535,  10.9996,  10.6730,  12.3507, -10.3809,   4.7243, -24.4257,
         -8.5909,   1.9024, -12.2744, -12.4751, -23.2778,  -2.0163,  25.8767,
         14.2108,  17.7691, -10.9204, -20.7335,   6.4560,  11.1615],
       grad_fn=<SelectBackward0>)

Loss after centering logits around "0":

      0/ 200000: 3.3221
  10000/ 200000: 2.1900
  20000/ 200000: 2.4196
  30000/ 200000: 2.6067
  40000/ 200000: 2.0601
  50000/ 200000: 2.4988
  60000/ 200000: 2.3902
  70000/ 200000: 2.1344
  80000/ 200000: 2.3369
  90000/ 200000: 2.1299
Decaying learning rate at the 100000 epoch.
 100000/ 200000: 1.8329
 110000/ 200000: 2.2053
 120000/ 200000: 1.8540
 130000/ 200000: 2.4566
 140000/ 200000: 2.1879
 150000/ 200000: 2.1118
 160000/ 200000: 1.8956
 170000/ 200000: 1.8644
 180000/ 200000: 2.0326
 190000/ 200000: 1.8417
train 2.069589138031006
val 2.1310746669769287
```

and the following plot gets generated:

![w13_lower_initial_loss.png](../assets/w13_lower_initial_loss.png?raw=true "w13_lower_initial_loss.png")

## Task 20

**Description:**

Run the network for $1$ epoch and visualize:

- the hidden layer pre-activation values (i.e. the logits in the hidden layer) using a histogram with $50$ bins;
- the output of the hidden layer using a histogram with $50$ bins.
- the amount of activations that are in the tails of $tanh$ (with a threshold of $.99$).

**Acceptance criteria:**

1. The test case passes.
2. One figure with three plots is produced.

**Test case:**

```console
python task20.py
```

The following plots get generated:

![w13_activations_initial.png](../assets/w13_activations_initial.png?raw=true "w13_activations_initial.png")

## Task 21

**Description:**

Fix the preactivations using hardcoded numbers and create the plots again. Play around with the values and try to get as close as possible to the plots you see in the test case.

Let the network train for the full amount of epochs, report the obtained losses and compare them to the previously seen ones.

**Acceptance criteria:**

1. The test case passes.
2. An analysis of the results is performed.

**Test case:**

```console
python task21.py
```

The following is printed on the console:

```console
      0/ 200000: 3.3135
  10000/ 200000: 2.1648
  20000/ 200000: 2.3061
  30000/ 200000: 2.4541
  40000/ 200000: 1.9787
  50000/ 200000: 2.2930
  60000/ 200000: 2.4232
  70000/ 200000: 2.0680
  80000/ 200000: 2.3095
  90000/ 200000: 2.1207
Decaying learning rate at the 100000 epoch.
 100000/ 200000: 1.8269
 110000/ 200000: 2.2045
 120000/ 200000: 1.9797
 130000/ 200000: 2.3946
 140000/ 200000: 2.1000
 150000/ 200000: 2.1948
 160000/ 200000: 1.8619
 170000/ 200000: 1.7809
 180000/ 200000: 1.9673
 190000/ 200000: 1.8295
train 2.0355966091156006
val 2.1026782989501953
```

And the following figure is generated:

![w13_activations_fixed.png](../assets/w13_activations_fixed.png?raw=true "w13_activations_fixed.png")

## Task 22

**Description:**

Fix the preactivations using Kaiming initialization. Ensure that you get (almost) the same loss as in the previous task.

**Acceptance criteria:**

1. The value of the loss is almost the same as in the previous task.
