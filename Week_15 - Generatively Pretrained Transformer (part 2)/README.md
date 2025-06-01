# 🎯 Goals for week 15

1. Implement a generatively pretrained transformer.
2. Practice writing high quality code:
   1. Easy to read.
   2. Safe from bugs.
   3. Ready for change.

## Task 1

**Description:**

Implement a single self-attention head. Use it to process the token encodings and produce the input for the final linear layer.

**Acceptance criteria:**

1. A self-attention head is implemented and used in the model.

**Test case:**

If the head size is equal to the size of the embeddings, the learning rate is `0.001` and the evaluation interval is `500`, then for `5000` epochs we should see something like this:

```console
python task01.py
```

```console
Epoch 0: avg train loss 4.2495 | avg val loss 4.2491
Epoch 500: avg train loss 2.7197 | avg val loss 2.7159
Epoch 1000: avg train loss 2.5196 | avg val loss 2.5141
Epoch 1500: avg train loss 2.4801 | avg val loss 2.4740
Epoch 2000: avg train loss 2.4395 | avg val loss 2.4639
Epoch 2500: avg train loss 2.4267 | avg val loss 2.4410
Epoch 3000: avg train loss 2.4123 | avg val loss 2.4284
Epoch 3500: avg train loss 2.4065 | avg val loss 2.4150
Epoch 4000: avg train loss 2.4017 | avg val loss 2.4197
Epoch 4500: avg train loss 2.3840 | avg val loss 2.4114

Sampling 500 tokens from a trained model: 
LIG Mit his ll,
I whosor pthe iing recrd aty, brey judr dilen vooniknisha bubrood f shes fo, to meat lto sthy thee hand esawaw hiat but the Be tviell plll rme? th cant soveteridusl ar trod hes lernase ms woreivengilveray' berle ant th weas, bren ongo thea loure MLRHANE:
UCous,
As nly ic'der TI boss oferer her: ale mal fours wir'sw the:, thou, we?

I were;
Had domerd th yowu hveros tcher os were helly thart I fo yorsm thal;
M'd wested.

Fougoor yu stst withy risrsthe alle grmerdie tiyos, pomu!
Lr
```

## Task 2

**Description:**

Implement multi-head attention.

**Acceptance criteria:**

1. Multiple heads of self-attention are used in the model.

**Test case:**

With $4$ heads we should get something like this:

```console
python task02.py
```

```console
Epoch 0: avg train loss 4.2200 | avg val loss 4.2214
Epoch 500: avg train loss 2.6426 | avg val loss 2.6414
Epoch 1000: avg train loss 2.4820 | avg val loss 2.4764
Epoch 1500: avg train loss 2.4307 | avg val loss 2.4226
Epoch 2000: avg train loss 2.3739 | avg val loss 2.3997
Epoch 2500: avg train loss 2.3458 | avg val loss 2.3582
Epoch 3000: avg train loss 2.3195 | avg val loss 2.3375
Epoch 3500: avg train loss 2.3085 | avg val loss 2.3168
Epoch 4000: avg train loss 2.2931 | avg val loss 2.3185
Epoch 4500: avg train loss 2.2731 | avg val loss 2.3049

Sampling 500 tokens from a trained model: 
LOUTK:
Ards:
Gsued-hoshald a thing rome thub, bres jedaf then vooniknis shembrood fourds fo, to meat lepas in thee hand eagwaw hiat but the Be tho lours lirme? thace his antere?
Whin: tend heselernase mat dinge to wheract beellennt ther

I,
Bof So Northea poure.
Lit mard, mat,
Asthen: horer Turboss of oe wher: ale malles us wir al the:
Erust, we?

Ildere;
Had dome dor'd isharveros tcher o corde helly thart I fory pum thave for weremy.

Fougtor yous starr:
O ris sot selle gre wore tivesr'poms!
La
```

## Task 3

**Description:**

Add the feed-forward layer that is applied after the self-attention heads in the `Transformer` paper. For now, keep this layer as a single linear layer.

**Acceptance criteria:**

1. The output of the self-attention block goes through a feed-forward layer.
2. The average validation loss is lower that the one in the previous task.

**Test case:**

If the hidden size of the newly added feed-forward linear layer is equal to the size of the embedding, then we should get something like the following:

```console
python task03.py
```

```console
Epoch 0: avg train loss 4.1875 | avg val loss 4.1858
Epoch 500: avg train loss 2.5998 | avg val loss 2.6074
Epoch 1000: avg train loss 2.4559 | avg val loss 2.4565
Epoch 1500: avg train loss 2.3926 | avg val loss 2.3881
Epoch 2000: avg train loss 2.3450 | avg val loss 2.3666
Epoch 2500: avg train loss 2.3103 | avg val loss 2.3251
Epoch 3000: avg train loss 2.2901 | avg val loss 2.3064
Epoch 3500: avg train loss 2.2710 | avg val loss 2.2848
Epoch 4000: avg train loss 2.2644 | avg val loss 2.2838
Epoch 4500: avg train loss 2.2275 | avg val loss 2.2645

Sampling 500 tokens from a trained model: 
BOMER:
But,' frocke poot oarty this nesece pry nosters
At orseo!

To nanom
And.'
Berw wolland canate the st ghe pingneft all anin ke of the in seeandrowne.

MAUCINARDWARCK:
ANGO my frand youl as you you fay.
I:
Wick.

NINILI:
Etfringque dall wis sosegh all, mis by darfitill wich slersill my tuss!

QUE RICA RI:
Your sesin apt seam mud, thof mided for eat homond Mid!

WALOU:
A an'd it'ply-
I land.

bly,
And upelllag- Vack aryou you iass din nind wilken buspis ait iin a ot.

UTIGORLO:
O:
Angioe it.
```

## Task 4

**Description:**

We are getting closer and closer to having all the parts that make up the decoder block of a decoder-only model (here we don't need cross-attention).

Let's create a class `Block` that encompasses the communication (multi-head self-attention subnetwork) and the computation (feed forward subnetwork).

Extend our network with three blocks, each with four attention heads.

**Acceptance criteria:**

1. A decoder block is implemented.
2. The network is extended to have three blocks, each with four attention heads.

**Test case:**

```console
python task04.py
```

```console
Epoch 0: avg train loss 4.1810 | avg val loss 4.1817
Epoch 500: avg train loss 3.0917 | avg val loss 3.1054
Epoch 1000: avg train loss 2.6925 | avg val loss 2.6808
Epoch 1500: avg train loss 2.5110 | avg val loss 2.5171
Epoch 2000: avg train loss 2.4418 | avg val loss 2.4424
Epoch 2500: avg train loss 2.4058 | avg val loss 2.3970
Epoch 3000: avg train loss 2.3453 | avg val loss 2.3633
Epoch 3500: avg train loss 2.3170 | avg val loss 2.3410
Epoch 4000: avg train loss 2.3074 | avg val loss 2.3208
Epoch 4500: avg train loss 2.2717 | avg val loss 2.2915

Sampling 500 tokens from a trained model: 
What and nis thimn my theared thall of cepreckn,
Of coume, and be
lapvits by we sakorghep
Thee us thardy ise deve ditleremang yof's, sto himavesan trour that ghe menm: hand Hretest dhis wive:
urore cease wim port, woass thee? Pmitorest;
Alm; fy, wear he wouls ofn!-
Gero; soe thesen, fere an thand:
Lit of thas.

EBEY REd pO, Rpeleme theld sosour thaun I mabe gty and;
That, bar te shefo, akow 'l.

INAL OLONARESY GARYIN:
KALIDUTITH:
Maron my ferily atess sralucunh, mard ake 'seals bars-tuut witief.
```

## Task 5

**Description:**

You'll notice that the pure addition of more decoder blocks did not really help increase the performance of our model. This is because our network is starting to get pretty deep and we know that deep networks often suffer from small gradient issues.

There are two optimizations that can help us alleviate these issues:

1. Skip connections.
2. Normalizations (batch or layer).

Let's try to fix to first apply skip-connections. Add these to our architecture as described in the Transformer paper. Use this opportunity to extend the feed-forward module to be an MLP, using the sizes described in the paper.

> Notice the improvements we get both in the validation loss and in the text generations 🥳!

**Acceptance criteria:**

1. Skip connections are implemented as described in the Transformer paper.
2. The feed-forwad module is extended to an MLP.

**Test case:**

```console
python task05.py
```

```console
Epoch 0: avg train loss 4.5651 | avg val loss 4.5756
Epoch 500: avg train loss 2.3911 | avg val loss 2.4035
Epoch 1000: avg train loss 2.2608 | avg val loss 2.2722
Epoch 1500: avg train loss 2.2093 | avg val loss 2.2400
Epoch 2000: avg train loss 2.1531 | avg val loss 2.1877
Epoch 2500: avg train loss 2.0892 | avg val loss 2.1526
Epoch 3000: avg train loss 2.0688 | avg val loss 2.1409
Epoch 3500: avg train loss 2.0579 | avg val loss 2.1359
Epoch 4000: avg train loss 2.0203 | avg val loss 2.0996
Epoch 4500: avg train loss 2.0061 | avg val loss 2.0771

Sampling 500 tokens from a trained model: 
Whit as fater;
't seg the med shall and bet crotursentlem,
And bethap will saye strore.

PrAUuluse I o you, aded,
Not aged not will poubth haves.
Secture hath prome machand-Hentern dueed with unor it sue wilt wet, what's mee?

LUCET:
And let ham we sain yours of thou knis derval herent bronend wellay, of that was well diph--
Hentient,
Thoses aree--un I mabesety sock: your bandeeds mire all then paws nemn.
Hess as now:
Kef SICINTET:
And my fartiomandss, till--le, many as Brient.

GLUCHABETH:
Sir.
```

## Task 6

**Description:**

Let's now add the second guard rail against gradient problems - normalizations. In the paper we can see that the authors use layer normalization. By looking at the paper, we can see that, originally, it is applied after the transformations. Recently, however, [a paper came out](https://arxiv.org/pdf/2002.04745) that popularized the "pre-ln Transformer". This is our regular Transformer but with the LN applied before the transformations. Add layer normalization to our network using this new style and check the results.

Having this implemented we now have a fully-functional decoder only Transformer! Congratulations for reaching this far 🥳!

**Acceptance criteria:**

1. Layer normalization is added in accordance with the guidelines outlined in the paper [On Layer Normalization in the Transformer Architecture](https://arxiv.org/pdf/2002.04745).

**Test case:**

The below output has the training loop modified a little bit by outputting the value of the loss at the very last epoch as well  .

```console
python task06.py
```

```console
Epoch 0: avg train loss 4.3126 | avg val loss 4.3167
Epoch 500: avg train loss 2.4067 | avg val loss 2.4157
Epoch 1000: avg train loss 2.2534 | avg val loss 2.2688
Epoch 1500: avg train loss 2.2049 | avg val loss 2.2293
Epoch 2000: avg train loss 2.1383 | avg val loss 2.1755
Epoch 2500: avg train loss 2.0827 | avg val loss 2.1540
Epoch 3000: avg train loss 2.0559 | avg val loss 2.1289
Epoch 3500: avg train loss 2.0428 | avg val loss 2.1231
Epoch 4000: avg train loss 2.0103 | avg val loss 2.0887
Epoch 4500: avg train loss 1.9941 | avg val loss 2.0686
Epoch 4999: avg train loss 1.9835 | avg val loss 2.0694

Sampling 500 tokens from a trained model: 
What as file,
Pronges theare distand he but crotue yet me, and by lapinisers we sproverel
The must I of a,
Kin it dity the not with posbrothateran theur that grectermach not shall he is will wulord cansely mare:
Hangess thee?
This menturel me here an he woulBe call; my dus he must mentriens no well my of that wren,
Thon prienpellient,
Thosost the lund.

COLAUS:
How wilt,
And when seir, all bold prese amnowerss as now:
Kser manase be on my fortheman ssostill your mare as Bose,
The whit unforting.
```

## Task 7

**Description:**

Let's scale up our network to see how much we can push this number $2.0694$.

Play around with the values of the following hyperparameters and try to lower the validation as much as possible. Because we're about the scale-up the model we might run into overfitting - feel free to add dropout layers as shown in the original Transformer paper.

Here're some hyperparameters (for inspiration) you might try tweaking:

- The batch size.
- The block size.
- The learning rate.
- The size of the embeddings.
- The number of attention heads in each decoder block.
- The number of decoder blocks.
- The dropout rate.

**Acceptance criteria:**

1. Hyperparameter optimization is performed.
2. Dropout is added.

**Test case:**

```console
python task07.py
```

```console
Epoch 0: avg train loss 4.3327 | avg val loss 4.3355
Epoch 500: avg train loss 2.0271 | avg val loss 2.0972
Epoch 1000: avg train loss 1.6219 | avg val loss 1.7953
Epoch 1500: avg train loss 1.4482 | avg val loss 1.6535
Epoch 2000: avg train loss 1.3516 | avg val loss 1.5824
Epoch 2500: avg train loss 1.2869 | avg val loss 1.5414
Epoch 3000: avg train loss 1.2343 | avg val loss 1.5184
Epoch 3500: avg train loss 1.1880 | avg val loss 1.5013
Epoch 4000: avg train loss 1.1516 | avg val loss 1.4868
Epoch 4500: avg train loss 1.1179 | avg val loss 1.4952
Epoch 4999: avg train loss 1.0845 | avg val loss 1.4894

Sampling 500 tokens from a trained model: 
VIRGIA:

VOLUToENTA:
Help, to thou the priest, I'll tell you fair,
If any it wis not mine abhorr'd;
That I might, because perilars again,
New, loving to go he accomptions,
Eaping roughly; to the caius bosom.

CORIOLANUS:
Nay, dispossible and spit with past while.

MENENIUS:
Mighter!
No, would he had not? What's he into this?
Is myself? I think your danger he has saved.

First Senator:
Well, how Case you not sleep in your worse?

First Murderer:
Must of you scanderly clearly she god:
Some comes y
```

## Task 8

**Description:**

Create a script that would use your best GPT yet to generate infinite Shakespeare. Running the script would start a text generation process that can only be stopped if the user quit the program.

**Acceptance criteria:**

1. A script is created that when run start producing Shakespearian-like text without stopping.

## Task 9

**Description:**

We saw that in more recent years there have been changes to the original Transformer architecture - ex. the position of the layer normalization module and the logic of the positional encodings. Read some newer Transformer papers and implement one additional feature or change that people seem to use. Does it improve the performance of your GPT?

**Acceptance criteria:**

1. Architecture is modified in such a way so as to obtain a lower validation loss.
2. Relevant papers are linked in the solution.
