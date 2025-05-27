# 🎯 Goals for week 14

1. Start to write out an implementation of a generatively pretrained transformer that can generate poems in the style of William Shakespeare.
2. Implement input embeddings and positional encodings of a generatively pretrained transformer.
3. Practice writing high quality code:
   1. Easy to read.
   2. Safe from bugs.
   3. Ready for change.

## Task 1

**Description:**

Explore the dataset we're going to use - the text file `shakespeare.txt`, present in our `DATA` folder:

1. Output the length of the dataset in characters.
2. Output the first $1000$ characters.

**Acceptance criteria:**

1. Output matches the one shown below.

**Test case:**

```console
python task01.py
```

```text
Number of characters: 1,115,394
First 1000 tokens:
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

```

## Task 2

**Description:**

Create the vocabulary. Output it and its size.

**Acceptance criteria:**

1. Output matches the one shown below.

**Test case:**

```console
python task02.py
```

```text
Vocabulary: 
 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
Vocabulary size: 65
```

## Task 3

**Description:**

Create a tokenizer that can:

- take text and return a list of token ids.
- take a list of token ids and return the text that they form.

**Acceptance criteria:**

1. Output matches the one shown below.

**Test case:**

(The word `therre` is intentionally written like that.)

```console
python task03.py
```

```text
Encoding the text "hi therre": [46, 47, 1, 58, 46, 43, 56, 56, 43]
Decoding the encoded version of the text "hi therre": hi therre
```

## Task 4

**Description:**

Tokenize the dataset and store the result in a tensor.

**Acceptance criteria:**

1. Output matches the one shown below.

**Test case:**

```console
python task04.py
```

```text
Shape of result: torch.Size([1115394])
Data type of result: torch.int64
The first 1000 tokens: tensor([18, 47, 56, 57, 58,  1, 15, 47, 58, 47, 64, 43, 52, 10,  0, 14, 43, 44,
        53, 56, 43,  1, 61, 43,  1, 54, 56, 53, 41, 43, 43, 42,  1, 39, 52, 63,
         1, 44, 59, 56, 58, 46, 43, 56,  6,  1, 46, 43, 39, 56,  1, 51, 43,  1,
        57, 54, 43, 39, 49,  8,  0,  0, 13, 50, 50, 10,  0, 31, 54, 43, 39, 49,
         6,  1, 57, 54, 43, 39, 49,  8,  0,  0, 18, 47, 56, 57, 58,  1, 15, 47,
        58, 47, 64, 43, 52, 10,  0, 37, 53, 59,  1, 39, 56, 43,  1, 39, 50, 50,
         1, 56, 43, 57, 53, 50, 60, 43, 42,  1, 56, 39, 58, 46, 43, 56,  1, 58,
        53,  1, 42, 47, 43,  1, 58, 46, 39, 52,  1, 58, 53,  1, 44, 39, 51, 47,
        57, 46, 12,  0,  0, 13, 50, 50, 10,  0, 30, 43, 57, 53, 50, 60, 43, 42,
         8,  1, 56, 43, 57, 53, 50, 60, 43, 42,  8,  0,  0, 18, 47, 56, 57, 58,
         1, 15, 47, 58, 47, 64, 43, 52, 10,  0, 18, 47, 56, 57, 58,  6,  1, 63,
        53, 59,  1, 49, 52, 53, 61,  1, 15, 39, 47, 59, 57,  1, 25, 39, 56, 41,
        47, 59, 57,  1, 47, 57,  1, 41, 46, 47, 43, 44,  1, 43, 52, 43, 51, 63,
         1, 58, 53,  1, 58, 46, 43,  1, 54, 43, 53, 54, 50, 43,  8,  0,  0, 13,
        50, 50, 10,  0, 35, 43,  1, 49, 52, 53, 61,  5, 58,  6,  1, 61, 43,  1,
        49, 52, 53, 61,  5, 58,  8,  0,  0, 18, 47, 56, 57, 58,  1, 15, 47, 58,
        47, 64, 43, 52, 10,  0, 24, 43, 58,  1, 59, 57,  1, 49, 47, 50, 50,  1,
        46, 47, 51,  6,  1, 39, 52, 42,  1, 61, 43,  5, 50, 50,  1, 46, 39, 60,
        43,  1, 41, 53, 56, 52,  1, 39, 58,  1, 53, 59, 56,  1, 53, 61, 52,  1,
        54, 56, 47, 41, 43,  8,  0, 21, 57,  5, 58,  1, 39,  1, 60, 43, 56, 42,
        47, 41, 58, 12,  0,  0, 13, 50, 50, 10,  0, 26, 53,  1, 51, 53, 56, 43,
         1, 58, 39, 50, 49, 47, 52, 45,  1, 53, 52,  5, 58, 11,  1, 50, 43, 58,
         1, 47, 58,  1, 40, 43,  1, 42, 53, 52, 43, 10,  1, 39, 61, 39, 63,  6,
         1, 39, 61, 39, 63,  2,  0,  0, 31, 43, 41, 53, 52, 42,  1, 15, 47, 58,
        47, 64, 43, 52, 10,  0, 27, 52, 43,  1, 61, 53, 56, 42,  6,  1, 45, 53,
        53, 42,  1, 41, 47, 58, 47, 64, 43, 52, 57,  8,  0,  0, 18, 47, 56, 57,
        58,  1, 15, 47, 58, 47, 64, 43, 52, 10,  0, 35, 43,  1, 39, 56, 43,  1,
        39, 41, 41, 53, 59, 52, 58, 43, 42,  1, 54, 53, 53, 56,  1, 41, 47, 58,
        47, 64, 43, 52, 57,  6,  1, 58, 46, 43,  1, 54, 39, 58, 56, 47, 41, 47,
        39, 52, 57,  1, 45, 53, 53, 42,  8,  0, 35, 46, 39, 58,  1, 39, 59, 58,
        46, 53, 56, 47, 58, 63,  1, 57, 59, 56, 44, 43, 47, 58, 57,  1, 53, 52,
         1, 61, 53, 59, 50, 42,  1, 56, 43, 50, 47, 43, 60, 43,  1, 59, 57, 10,
         1, 47, 44,  1, 58, 46, 43, 63,  0, 61, 53, 59, 50, 42,  1, 63, 47, 43,
        50, 42,  1, 59, 57,  1, 40, 59, 58,  1, 58, 46, 43,  1, 57, 59, 54, 43,
        56, 44, 50, 59, 47, 58, 63,  6,  1, 61, 46, 47, 50, 43,  1, 47, 58,  1,
        61, 43, 56, 43,  0, 61, 46, 53, 50, 43, 57, 53, 51, 43,  6,  1, 61, 43,
         1, 51, 47, 45, 46, 58,  1, 45, 59, 43, 57, 57,  1, 58, 46, 43, 63,  1,
        56, 43, 50, 47, 43, 60, 43, 42,  1, 59, 57,  1, 46, 59, 51, 39, 52, 43,
        50, 63, 11,  0, 40, 59, 58,  1, 58, 46, 43, 63,  1, 58, 46, 47, 52, 49,
         1, 61, 43,  1, 39, 56, 43,  1, 58, 53, 53,  1, 42, 43, 39, 56, 10,  1,
        58, 46, 43,  1, 50, 43, 39, 52, 52, 43, 57, 57,  1, 58, 46, 39, 58,  0,
        39, 44, 44, 50, 47, 41, 58, 57,  1, 59, 57,  6,  1, 58, 46, 43,  1, 53,
        40, 48, 43, 41, 58,  1, 53, 44,  1, 53, 59, 56,  1, 51, 47, 57, 43, 56,
        63,  6,  1, 47, 57,  1, 39, 57,  1, 39, 52,  0, 47, 52, 60, 43, 52, 58,
        53, 56, 63,  1, 58, 53,  1, 54, 39, 56, 58, 47, 41, 59, 50, 39, 56, 47,
        57, 43,  1, 58, 46, 43, 47, 56,  1, 39, 40, 59, 52, 42, 39, 52, 41, 43,
        11,  1, 53, 59, 56,  0, 57, 59, 44, 44, 43, 56, 39, 52, 41, 43,  1, 47,
        57,  1, 39,  1, 45, 39, 47, 52,  1, 58, 53,  1, 58, 46, 43, 51,  1, 24,
        43, 58,  1, 59, 57,  1, 56, 43, 60, 43, 52, 45, 43,  1, 58, 46, 47, 57,
         1, 61, 47, 58, 46,  0, 53, 59, 56,  1, 54, 47, 49, 43, 57,  6,  1, 43,
        56, 43,  1, 61, 43,  1, 40, 43, 41, 53, 51, 43,  1, 56, 39, 49, 43, 57,
        10,  1, 44, 53, 56,  1, 58, 46, 43,  1, 45, 53, 42, 57,  1, 49, 52, 53,
        61,  1, 21,  0, 57, 54, 43, 39, 49,  1, 58, 46, 47, 57,  1, 47, 52,  1,
        46, 59, 52, 45, 43, 56,  1, 44, 53, 56,  1, 40, 56, 43, 39, 42,  6,  1,
        52, 53, 58,  1, 47, 52,  1, 58, 46, 47, 56, 57, 58,  1, 44, 53, 56,  1,
        56, 43, 60, 43, 52, 45, 43,  8,  0,  0])
```

## Task 5

**Description:**

Split the dataset into `90%` training and `10%` testing sets. Output their shapes.

**Acceptance criteria:**

1. The gaps in the test case are filled-in.
2. The split is done in a way that takes into account the nature of the text data.

**Test case:**

```console
python task05.py
```

```text
Shape of dataset for training : <your numbers here>
Shape of dataset for testing: <your numbers here>
```

## Task 6

**Description:**

Let's define our sequence length (equiv. `block_size`) to be $8$. Output all the training samples formed by the first $9$ tokens.

**Acceptance criteria:**

1. Output matches the one shown below.

**Test case:**

```console
python task06.py
```

```text
Sample 1: given tensor([18]), predict 47
Sample 2: given tensor([18, 47]), predict 56
Sample 3: given tensor([18, 47, 56]), predict 57
Sample 4: given tensor([18, 47, 56, 57]), predict 58
Sample 5: given tensor([18, 47, 56, 57, 58]), predict 1
Sample 6: given tensor([18, 47, 56, 57, 58,  1]), predict 15
Sample 7: given tensor([18, 47, 56, 57, 58,  1, 15]), predict 47
Sample 8: given tensor([18, 47, 56, 57, 58,  1, 15, 47]), predict 58
```

## Task 7

**Description:**

Let's define our batch size to be $4$. Output the independent sequences that the model will process in parallel for a randomly obtained batch.

Use a seed of $42$.

**Acceptance criteria:**

1. Output matches the one shown below.

**Test case:**

```console
python task07.py
```

```text
Inputs:
torch.Size([4, 8])
tensor([[57,  1, 46, 47, 57,  1, 50, 53],
        [ 1, 58, 46, 43, 56, 43,  1, 41],
        [17, 26, 15, 17, 10,  0, 32, 53],
        [57, 58,  6,  1, 61, 47, 58, 46]])

Targets:
torch.Size([4, 8])
tensor([[ 1, 46, 47, 57,  1, 50, 53, 60],
        [58, 46, 43, 56, 43,  1, 41, 39],
        [26, 15, 17, 10,  0, 32, 53,  1],
        [58,  6,  1, 61, 47, 58, 46,  0]])
----
Sample 1: given [57], predict 1
Sample 2: given [57, 1], predict 46
Sample 3: given [57, 1, 46], predict 47
Sample 4: given [57, 1, 46, 47], predict 57
Sample 5: given [57, 1, 46, 47, 57], predict 1
Sample 6: given [57, 1, 46, 47, 57, 1], predict 50
Sample 7: given [57, 1, 46, 47, 57, 1, 50], predict 53
Sample 8: given [57, 1, 46, 47, 57, 1, 50, 53], predict 60
Sample 1: given [1], predict 58
Sample 2: given [1, 58], predict 46
Sample 3: given [1, 58, 46], predict 43
Sample 4: given [1, 58, 46, 43], predict 56
Sample 5: given [1, 58, 46, 43, 56], predict 43
Sample 6: given [1, 58, 46, 43, 56, 43], predict 1
Sample 7: given [1, 58, 46, 43, 56, 43, 1], predict 41
Sample 8: given [1, 58, 46, 43, 56, 43, 1, 41], predict 39
Sample 1: given [17], predict 26
Sample 2: given [17, 26], predict 15
Sample 3: given [17, 26, 15], predict 17
Sample 4: given [17, 26, 15, 17], predict 10
Sample 5: given [17, 26, 15, 17, 10], predict 0
Sample 6: given [17, 26, 15, 17, 10, 0], predict 32
Sample 7: given [17, 26, 15, 17, 10, 0, 32], predict 53
Sample 8: given [17, 26, 15, 17, 10, 0, 32, 53], predict 1
Sample 1: given [57], predict 58
Sample 2: given [57, 58], predict 6
Sample 3: given [57, 58, 6], predict 1
Sample 4: given [57, 58, 6, 1], predict 61
Sample 5: given [57, 58, 6, 1, 61], predict 47
Sample 6: given [57, 58, 6, 1, 61, 47], predict 58
Sample 7: given [57, 58, 6, 1, 61, 47, 58], predict 46
Sample 8: given [57, 58, 6, 1, 61, 47, 58, 46], predict 0
```

## Task 8

**Description:**

Create an $n$gram language model as discussed in `notes.md`. Pass the above generated batch to it and output:

- the shape of the logits.
- the value of the loss.

Answer the following question in a comment: *Compare the value of the loss with the baseline model - what can you conclude?*.

**Acceptance criteria:**

1. Output matches the one shown below.
2. A comment is present with an answer to the question in the description.

**Test case:**

```console
python task08.py
```

```text
Shape of logits: torch.Size([32, 65])
Loss: 4.724130153656006
```

## Task 9

**Description:**

Implement a functionality to generate from the model until a limit `max_new_tokens` is reached.

**Acceptance criteria:**

1. The model can generate new text for any arbitrary length, specified by the user of the model.

**Test case:**

```console
python task09.py
```

```text
Sampling 100 tokens from an untrained model: 
cfYCDRUZsYBsA?Y?vgB!ZWOEiAoezL:q&Avufr?gSGdWrp&Bxt-R?wo'TYhBChdIC-RDaRmEGENyouVg'UjyQNyQSpZUVeN:BZqh
```

## Task 10

**Description:**

Change the batch size to $32$ and train the model for $N$ epochs (you decide what $N$ should be) and output the new generations.

**Acceptance criteria:**

1. The generated text is a little bit more Shakespeare-like than it was in the previous task.

**Test case:**

```console
python task10.py
```

```text
Final loss: 2.543416976928711
Sampling 500 tokens from a trained model: 
HY:
R E:
NT:
Fowote e fofirelavese dVds wour yolo ayokie amblesbatave inouther hes molofuroriYO:
TheQUDUThe chas.
F lisen tabr:
LI mus nk,
A: al l ayo cenghe's therinvar,
TEsen ithawaneit at iswinerainy atsomo clour pad d wikn h,
HYy my Tholes: it GBy ke m vilou xthazinderand llo chee lond Cld this lisesule wars, tirofof wnofan
Rou cthe p.

By hat celis ire m, aksthethe aur withAR wotoot.
Toy:me, of Ithed; bo r:
DWAy celowinoourne, llonthavelller:f fowhilong bert irw:
I m;
ADWhit hor hy t I nd, 
```

## Task 11

**Description:**

Let's add a function for evaluating the model during training.

One evaluation loop would correspond to obtaining a batch of randomly sampled `xs` and `ys` from a certain data split and calculating the loss of the model.

Given that we're in the training loop, every `eval_interval` epochs, for the training and validation splits, perform `eval_iters` evaluation loops and report the average training and validation losses.

**Acceptance criteria:**

1. New functionality is introduced that performs `eval_iters` evaluation loops every `eval_interval` epochs.

**Test case:**

If we set `eval_interval=300` and `eval_iters=200`, we should get something similar to:

```console
python task11.py
```

```text
Epoch 0: avg train loss 4.7627 | avg val loss 4.7633
Epoch 300: avg train loss 2.8415 | avg val loss 2.8635
Epoch 600: avg train loss 2.5515 | avg val loss 2.5881
Epoch 900: avg train loss 2.4970 | avg val loss 2.5305
Epoch 1200: avg train loss 2.4908 | avg val loss 2.5049
Epoch 1500: avg train loss 2.4716 | avg val loss 2.5028
Epoch 1800: avg train loss 2.4557 | avg val loss 2.4971
Epoch 2100: avg train loss 2.4725 | avg val loss 2.4942
Epoch 2400: avg train loss 2.4637 | avg val loss 2.4956
Epoch 2700: avg train loss 2.4644 | avg val loss 2.4894

Sampling 500 tokens from a trained model: 
I
O: asafite tishere othut mod thand he, preckn,

Henthif o--wishelapinisercald burorean,
TANCluprt, o y inde Miced tlat mang yofowhas
brothangean t.


D hathapr me machan fl Cousu d iere--sthurore cense wn cr me, woass t avest t, metha'elm; hy?


TINGS:
larBRGLONGof ind s heve she, ENVI t, ndowe:
LUSio be's lore we, ba ph, Rpefoue t he ososoureealunghimyor g y stckis thed; bere s mire akns t
Thanesa amnfod bs as Y w:
K:
Tharease VOLon my feristhatessotrilorule, maripakn 'sealanks, g duthintide.
```

## Task 12

**Description:**

Currently, we have a model that is a single layer. Our goal is to turn this model into a `Decoder` from the `Transformer` architecture.

Let's start moving in that direction by:

1. Adding a hyperparameter for the size of the embeddings and encodings.
2. Adding positional encodings.
3. Adding a linear layer that will take in the token encodings and produce the final output.

In more recent years the logic of the positional encodings has been simplified to just using another set of embedding weights. Let's follow this approach here - implement the positional encodings as an embedding table.

After adding the new layers repeat the experiment from the previous task and output the new results.

Having achieved this we're ready to continue with the self-attention blocks of the decoder in the next exercises!

**Acceptance criteria:**

1. The size of the input embeddings and positional encodings is configurable.
2. Positional encoding is implemented.
3. A linear layer is added to transform the hidden layer outputs into the possible classes.

**Test case:**

If we set the size of the embeddings and encodings to $32$ then we should get something similar to:

```console
python task12.py
```

```text
Epoch 0: avg train loss 4.4609 | avg val loss 4.4476
Epoch 300: avg train loss 2.5513 | avg val loss 2.5633
Epoch 600: avg train loss 2.5228 | avg val loss 2.5578
Epoch 900: avg train loss 2.5039 | avg val loss 2.5354
Epoch 1200: avg train loss 2.5051 | avg val loss 2.5157
Epoch 1500: avg train loss 2.4971 | avg val loss 2.5354
Epoch 1800: avg train loss 2.4783 | avg val loss 2.5142
Epoch 2100: avg train loss 2.5051 | avg val loss 2.5297
Epoch 2400: avg train loss 2.4897 | avg val loss 2.5205
Epoch 2700: avg train loss 2.4948 | avg val loss 2.5162

Sampling 500 tokens from a trained model: 
Houoosomousseve tt are are ee avoue gemer

If se t awimoucr:
TE:
F tees on t ch seyoworeth Hinerd ury e ms ld peareacagongere we asentha s; we ws is bpachid withissot th lowe m d,
Te t, wem her D caimele thone?
AMa d wimy acatre reain ngllouadruld ff araned ber fouly!
Myonow il k y quinovereave areal ouco w.
A:
Dice;
NR:


womelo m,
CSAno ge COrefafou's t

S:
NGAn se, t.
be:
he mero crd Bour tietegotwize tr wideaim,
YCo beg towiord
I tecure,
L: je y, t athe sstitango;

Timowerutherakint wetoound
```
