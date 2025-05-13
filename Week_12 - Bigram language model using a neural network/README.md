# 🎯 Goals for week 12

1. Implement a character-level **bigram language model** using a neural network (instead of counting/probabilities).
2. Practice writing high quality code:
   1. Easy to read.
   2. Safe from bugs.
   3. Ready for change.

## Task 1

**Description:**

Create the training set for the neural network from the `names.txt` file.

**Acceptance criteria:**

1. Two tensors are created - `xs` and `ys`. The tensor `xs` holds the index (position in the alphabet) of the character and `ys` holds the index of the character that should follow it.
2. The dataset is comprised of integer values.
3. The solution comprises solely of custom code - no layers or classes from `torch` should be used.

**Test case:**

For the word `emma`, the dataset should contain `5` separate input examples:

`xs = tensor([0, 5, 13, 13, 1])`

and

`ys = tensor([5, 13, 13, 1, 0])`

## Task 2

**Description:**

One hot encode `xs` and `ys`.

**Acceptance criteria:**

1. Each subtensor has a length of `27`.
2. The data type of both datasets is `float`.
3. The solution comprises solely of custom code - no layers from `torch.nn` should be used.

**Test case:**

The encoding for the word `emma` if printed should look like this:

```python
tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
```

The encoding for the word `emma` if plotted using `plt.imshow` should look like this:

![xenc](../assets/w12_xenc.png?raw=true "xenc.png")

## Task 3

**Description:**

Create a single neuron without a bias and an activation function and pass the training examples through it.

**Acceptance criteria:**

1. The above architecture is implemented.
2. The shape of the output is `(n, 1)` where `n` is the number of examples passed.
3. The solution comprises solely of custom code - no layers from `torch.nn` should be used.

**Test case:**

If the `5` examples, generated from the encoded version of the word `emma`, are passed through the model, the output should be similar to:

```python
tensor([[-0.4947],
        [-0.3425],
        [-0.1272],
        [-0.1272],
        [ 0.1293]])
```

## Task 4

**Description:**

Create the neural network with `27` neurons in the hidden layer that we discussed in `notes.md`.

**Acceptance criteria:**

1. The test cases pass.
2. The solution comprises solely of custom code - no layers from `torch.nn` should be used.

**Test cases:**

If the word `emma` is passed through it and the output variable is called `probs`, then:

```python
print(f'{probs.shape=}') # ensure every character gets mapped to every possible character
print(f'{probs[0].sum()=}') # ensure a randomly picked row sums up to 1
```

should output:

```console
probs.shape=torch.Size([5, 27])
probs[0].sum()=tensor(1.)
```

## Task 5

**Description:**

Create a more detailed process to track what the network does. For each bigram example output:

- the bigram and its indices;
- the index that is fed into the neural net;
- the full vector with probabilities;
- the label that actually follows the passed character;
- the probability assigned by the neural network to the label;
- the log likelihood of the probability;
- the negative log likelihood of the probability.

After all bigrams are passed, output the **average negative log likelihood**, i.e. the **loss** of our model.

Note that running the script multiple times, produces different values for the loss function - some being good, some being bad.

**Acceptance criteria:**

1. The test cases for the word `emma` passes.
2. All print statements match!
3. The solution comprises solely of custom code - no layers from `torch.nn` should be used.

**Test cases:**

If the word `emma` is passed, then the output should be something similar to:

```console
------
bigram example 1: .e (indexes 0,5)
input to the neural net: 0
output probabilities from the neural net: tensor([0.0100, 0.0190, 0.0464, 0.0201, 0.0149, 0.0206, 0.0036, 0.0135, 0.0637,
        0.0196, 0.0195, 0.0668, 0.0659, 0.0065, 0.0396, 0.0209, 0.0511, 0.0398,
        0.0712, 0.0444, 0.0127, 0.0025, 0.0557, 0.0154, 0.1192, 0.1191, 0.0183])
label (actual next character): 5
probability assigned by the neural net to the label: 0.020570652559399605
log likelihood: -3.88388991355896
negative log likelihood: 3.88388991355896
------
bigram example 2: em (indexes 5,13)
input to the neural net: 5
output probabilities from the neural net: tensor([0.0387, 0.0138, 0.0556, 0.0107, 0.0225, 0.0385, 0.0421, 0.0222, 0.0066,
        0.0163, 0.0768, 0.0035, 0.0565, 0.0327, 0.0145, 0.0618, 0.0952, 0.0092,
        0.0253, 0.0319, 0.0219, 0.0229, 0.0441, 0.0100, 0.1636, 0.0050, 0.0581])
label (actual next character): 13
probability assigned by the neural net to the label: 0.03271116316318512
log likelihood: -3.420038938522339
negative log likelihood: 3.420038938522339
------
bigram example 3: mm (indexes 13,13)
input to the neural net: 13
output probabilities from the neural net: tensor([0.0176, 0.0080, 0.0357, 0.0184, 0.0614, 0.0115, 0.2263, 0.0169, 0.1493,
        0.0145, 0.0405, 0.0043, 0.1459, 0.0602, 0.0109, 0.0058, 0.0088, 0.0043,
        0.0102, 0.0107, 0.0161, 0.0123, 0.0007, 0.0422, 0.0309, 0.0107, 0.0258])
label (actual next character): 13
probability assigned by the neural net to the label: 0.06015074625611305
log likelihood: -2.810901403427124
negative log likelihood: 2.810901403427124
------
bigram example 4: ma (indexes 13,1)
input to the neural net: 13
output probabilities from the neural net: tensor([0.0176, 0.0080, 0.0357, 0.0184, 0.0614, 0.0115, 0.2263, 0.0169, 0.1493,
        0.0145, 0.0405, 0.0043, 0.1459, 0.0602, 0.0109, 0.0058, 0.0088, 0.0043,
        0.0102, 0.0107, 0.0161, 0.0123, 0.0007, 0.0422, 0.0309, 0.0107, 0.0258])
label (actual next character): 1
probability assigned by the neural net to the label: 0.008036182262003422
log likelihood: -4.823801040649414
negative log likelihood: 4.823801040649414
------
bigram example 5: a. (indexes 1,0)
input to the neural net: 1
output probabilities from the neural net: tensor([0.0298, 0.0453, 0.1351, 0.0278, 0.0369, 0.0098, 0.0403, 0.0265, 0.0151,
        0.0206, 0.1024, 0.0123, 0.0540, 0.0571, 0.0094, 0.0479, 0.0218, 0.0408,
        0.0414, 0.0301, 0.0203, 0.0802, 0.0122, 0.0032, 0.0536, 0.0132, 0.0130])
label (actual next character): 0
probability assigned by the neural net to the label: 0.029797296971082687
log likelihood: -3.5133376121520996
negative log likelihood: 3.5133376121520996
======
average negative log likelihood, i.e. loss = 3.6903939247131348
```

## Task 6

**Description:**

Running the previous script multiple times to get a good model is not scalable and optimal.

Implement the training process using automatic gradient-based optimization. Use the analogy from the training process for the manually created `MLP`. Train for `100` epochs using a learning rate of `70`. To initialize the weights use the following generator: `g = torch.Generator().manual_seed(2147483647)`.

> **Note**: Values of the learning rate above `0.1` are extremely uncommon, however, because this task is very easy and our network is extremely small, we require either a high number of epochs (above `1000`) and a rather small learning rate, or lower number of epochs, but very high learning rate.

Output the number of examples used for training and the loss on each epoch. After training, plot the obtained losses against the epoch to see how they progress.

**Acceptance criteria:**

1. The test case passes.
2. The solution comprises solely of custom code - no layers from `torch.nn` should be used.

**Test cases:**

```console
python task06.py
```

```console
Training with 228146 examples.

Epoch [1/100]: 3.758953809738159
Epoch [2/100]: 3.2562801837921143
Epoch [3/100]: 3.0330348014831543
Epoch [4/100]: 2.9046101570129395
Epoch [5/100]: 2.8223724365234375
Epoch [6/100]: 2.765528678894043
Epoch [7/100]: 2.723264217376709
Epoch [8/100]: 2.6902756690979004
Epoch [9/100]: 2.6636600494384766
Epoch [10/100]: 2.641768217086792
Epoch [11/100]: 2.623516082763672
Epoch [12/100]: 2.6081504821777344
Epoch [13/100]: 2.595089912414551
Epoch [14/100]: 2.583888053894043
Epoch [15/100]: 2.5741915702819824
Epoch [16/100]: 2.5657260417938232
Epoch [17/100]: 2.55827260017395
Epoch [18/100]: 2.5516624450683594
Epoch [19/100]: 2.5457589626312256
Epoch [20/100]: 2.5404551029205322
Epoch [21/100]: 2.5356643199920654
Epoch [22/100]: 2.531318187713623
Epoch [23/100]: 2.527360439300537
Epoch [24/100]: 2.5237438678741455
Epoch [25/100]: 2.520430088043213
Epoch [26/100]: 2.517385482788086
Epoch [27/100]: 2.5145812034606934
Epoch [28/100]: 2.5119922161102295
Epoch [29/100]: 2.509596824645996
Epoch [30/100]: 2.507375717163086
Epoch [31/100]: 2.505312204360962
Epoch [32/100]: 2.503390312194824
Epoch [33/100]: 2.5015974044799805
Epoch [34/100]: 2.4999213218688965
Epoch [35/100]: 2.4983513355255127
Epoch [36/100]: 2.496878147125244
Epoch [37/100]: 2.4954938888549805
Epoch [38/100]: 2.494190216064453
Epoch [39/100]: 2.4929606914520264
Epoch [40/100]: 2.4917991161346436
Epoch [41/100]: 2.4907002449035645
Epoch [42/100]: 2.489659547805786
Epoch [43/100]: 2.4886720180511475
Epoch [44/100]: 2.487734079360962
Epoch [45/100]: 2.4868414402008057
Epoch [46/100]: 2.4859917163848877
Epoch [47/100]: 2.4851815700531006
Epoch [48/100]: 2.484407901763916
Epoch [49/100]: 2.4836692810058594
Epoch [50/100]: 2.482962131500244
Epoch [51/100]: 2.482285499572754
Epoch [52/100]: 2.4816370010375977
Epoch [53/100]: 2.481015205383301
Epoch [54/100]: 2.4804182052612305
Epoch [55/100]: 2.479844808578491
Epoch [56/100]: 2.4792933464050293
Epoch [57/100]: 2.4787635803222656
Epoch [58/100]: 2.47825288772583
Epoch [59/100]: 2.4777615070343018
Epoch [60/100]: 2.477287530899048
Epoch [61/100]: 2.4768307209014893
Epoch [62/100]: 2.4763901233673096
Epoch [63/100]: 2.475964307785034
Epoch [64/100]: 2.475553274154663
Epoch [65/100]: 2.47515606880188
Epoch [66/100]: 2.4747719764709473
Epoch [67/100]: 2.474400520324707
Epoch [68/100]: 2.474040985107422
Epoch [69/100]: 2.4736931324005127
Epoch [70/100]: 2.473355770111084
Epoch [71/100]: 2.473029375076294
Epoch [72/100]: 2.472712993621826
Epoch [73/100]: 2.4724061489105225
Epoch [74/100]: 2.4721083641052246
Epoch [75/100]: 2.4718194007873535
Epoch [76/100]: 2.47153902053833
Epoch [77/100]: 2.471266746520996
Epoch [78/100]: 2.4710023403167725
Epoch [79/100]: 2.470745325088501
Epoch [80/100]: 2.4704957008361816
Epoch [81/100]: 2.470252513885498
Epoch [82/100]: 2.4700164794921875
Epoch [83/100]: 2.4697868824005127
Epoch [84/100]: 2.4695630073547363
Epoch [85/100]: 2.4693453311920166
Epoch [86/100]: 2.469133138656616
Epoch [87/100]: 2.4689266681671143
Epoch [88/100]: 2.4687254428863525
Epoch [89/100]: 2.468529462814331
Epoch [90/100]: 2.4683382511138916
Epoch [91/100]: 2.4681520462036133
Epoch [92/100]: 2.467970132827759
Epoch [93/100]: 2.467792510986328
Epoch [94/100]: 2.4676194190979004
Epoch [95/100]: 2.4674506187438965
Epoch [96/100]: 2.467285394668579
Epoch [97/100]: 2.4671242237091064
Epoch [98/100]: 2.4669668674468994
Epoch [99/100]: 2.466813087463379
Epoch [100/100]: 2.4666624069213867
```

and the following diagram is produced:

![nn_output](../assets/w12_nn_output.png?raw=true "nn_output.png")

## Task 7

**Description:**

Add regularization when calculating the loss. Use $\lambda=0.01$. Output the obtained loss for each epoch.

**Acceptance criteria:**

1. The test case passes.
2. Regularization is added.
3. The solution comprises solely of custom code - no layers from `torch.nn` should be used.

**Test cases:**

```console
python task07.py"
```

```console
Training with 228146 examples and regularization.

Epoch [1/100]: 3.7686190605163574
Epoch [2/100]: 3.2635040283203125
Epoch [3/100]: 3.039781332015991
Epoch [4/100]: 2.9113266468048096
Epoch [5/100]: 2.8292152881622314
Epoch [6/100]: 2.7725422382354736
Epoch [7/100]: 2.7304797172546387
Epoch [8/100]: 2.6977224349975586
Epoch [9/100]: 2.671365261077881
Epoch [10/100]: 2.649750232696533
Epoch [11/100]: 2.6317856311798096
Epoch [12/100]: 2.616708755493164
Epoch [13/100]: 2.6039340496063232
Epoch [14/100]: 2.5930120944976807
Epoch [15/100]: 2.583587169647217
Epoch [16/100]: 2.5753843784332275
Epoch [17/100]: 2.5681874752044678
Epoch [18/100]: 2.5618255138397217
Epoch [19/100]: 2.5561628341674805
Epoch [20/100]: 2.551093816757202
Epoch [21/100]: 2.5465316772460938
Epoch [22/100]: 2.5424084663391113
Epoch [23/100]: 2.538667678833008
Epoch [24/100]: 2.5352628231048584
Epoch [25/100]: 2.5321547985076904
Epoch [26/100]: 2.5293099880218506
Epoch [27/100]: 2.5266997814178467
Epoch [28/100]: 2.5242996215820312
Epoch [29/100]: 2.522087574005127
Epoch [30/100]: 2.5200443267822266
Epoch [31/100]: 2.5181522369384766
Epoch [32/100]: 2.5163979530334473
Epoch [33/100]: 2.5147674083709717
Epoch [34/100]: 2.5132486820220947
Epoch [35/100]: 2.5118322372436523
Epoch [36/100]: 2.5105080604553223
Epoch [37/100]: 2.509268045425415
Epoch [38/100]: 2.5081052780151367
Epoch [39/100]: 2.5070130825042725
Epoch [40/100]: 2.5059854984283447
Epoch [41/100]: 2.505017042160034
Epoch [42/100]: 2.504103660583496
Epoch [43/100]: 2.5032405853271484
Epoch [44/100]: 2.5024237632751465
Epoch [45/100]: 2.501650333404541
Epoch [46/100]: 2.5009162425994873
Epoch [47/100]: 2.5002198219299316
Epoch [48/100]: 2.4995572566986084
Epoch [49/100]: 2.498927354812622
Epoch [50/100]: 2.4983270168304443
Epoch [51/100]: 2.4977548122406006
Epoch [52/100]: 2.497209310531616
Epoch [53/100]: 2.496687889099121
Epoch [54/100]: 2.496190071105957
Epoch [55/100]: 2.495713472366333
Epoch [56/100]: 2.495257616043091
Epoch [57/100]: 2.4948208332061768
Epoch [58/100]: 2.4944026470184326
Epoch [59/100]: 2.4940011501312256
Epoch [60/100]: 2.4936158657073975
Epoch [61/100]: 2.493246555328369
Epoch [62/100]: 2.492891550064087
Epoch [63/100]: 2.4925506114959717
Epoch [64/100]: 2.492222547531128
Epoch [65/100]: 2.4919071197509766
Epoch [66/100]: 2.491603374481201
Epoch [67/100]: 2.4913108348846436
Epoch [68/100]: 2.4910295009613037
Epoch [69/100]: 2.490757942199707
Epoch [70/100]: 2.4904966354370117
Epoch [71/100]: 2.4902443885803223
Epoch [72/100]: 2.4900007247924805
Epoch [73/100]: 2.4897663593292236
Epoch [74/100]: 2.489539384841919
Epoch [75/100]: 2.4893205165863037
Epoch [76/100]: 2.4891090393066406
Epoch [77/100]: 2.4889047145843506
Epoch [78/100]: 2.4887068271636963
Epoch [79/100]: 2.488515853881836
Epoch [80/100]: 2.4883310794830322
Epoch [81/100]: 2.488152027130127
Epoch [82/100]: 2.487978935241699
Epoch [83/100]: 2.4878110885620117
Epoch [84/100]: 2.4876489639282227
Epoch [85/100]: 2.4874916076660156
Epoch [86/100]: 2.4873390197753906
Epoch [87/100]: 2.4871909618377686
Epoch [88/100]: 2.4870479106903076
Epoch [89/100]: 2.4869086742401123
Epoch [90/100]: 2.48677396774292
Epoch [91/100]: 2.4866433143615723
Epoch [92/100]: 2.486515998840332
Epoch [93/100]: 2.4863927364349365
Epoch [94/100]: 2.4862730503082275
Epoch [95/100]: 2.486156463623047
Epoch [96/100]: 2.4860434532165527
Epoch [97/100]: 2.485933542251587
Epoch [98/100]: 2.4858264923095703
Epoch [99/100]: 2.485722780227661
Epoch [100/100]: 2.485621929168701
```

## Task 8

**Description:**

Generate `10` new words with the model. What do you notice?

**Acceptance criteria:**

1. The test case passes.
2. The solution comprises solely of custom code - no layers from `torch.nn` should be used.

**Test cases:**

```console
python task08.py"
```

```console
['ya', 'syahavilin', 'dleekahmangonya', 'tryahe', 'chen', 'ena', 'da', 'amiiae', 'a', 'keles']
```

The output above is the same as the first 10 words that were generated using the statistical approach.

## Task 9

**Description:**

Train a trigram language model, i.e. take two characters as an input to predict the 3rd one. Evaluate the loss. Did it improve over a bigram model?

**Acceptance criteria:**

1. A trigram model is created.
2. Sampling from a trained model is performed.
3. The loss of the trigram model is outputted.
4. An analysis is performed on the results.

## Task 10

**Description:**

Split up the dataset randomly into `80%` train set, `10%` dev/validation set, `10%` test set. Train the bigram and trigram models only on the training set and perform an evaluation on the validation set at the end of every epoch. Use the validation set to tune the strength of smoothing (or regularization) for the trigram model - i.e. try many possibilities via `a grid search` and see which one works best based on the validation loss.

Evaluate on the test sets after all the epochs. Plot the training loss against the validation loss over the epochs. Output the test loss. What can you see?

**Acceptance criteria:**

1. The full dataset is broken down into three sets - for training, for validation and for testing.
2. Hyperparameter parameter optimization is performed on the validation set.
3. A plot is produced with the training vs validation loss.
4. An analysis is performed on the results.
