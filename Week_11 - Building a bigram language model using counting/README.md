# 🎯 Goals for week 11

1. Implement a character-level **bigram language model** that can generate human-like names using only counting.
2. Practice writing high quality code:
   1. Easy to read.
   2. Safe from bugs.
   3. Ready for change.

## Task 1

**Description:**

Explore the dataset we're going to use - the text file `names.txt`, present in our `DATA` folder:

1. Output the first 10 names.
2. Output the total number of words.
3. Output the length of the shortest word.
4. Output the length of the longest word.

**Test case:**

```console
python task01.py
```

```console
['emma', 'olivia', 'ava', 'isabella', 'sophia', 'charlotte', 'mia', 'amelia', 'harper', 'evelyn']
Total number of words: 32,033.
Length of shortest word: 2.
Length of longest word: 15.
```

## Task 2

**Description:**

Output the bigrams generated for the first word.

**Test case:**

```console
python task02.py
```

```console
e m
m m
m a
```

## Task 3

**Description:**

Add the special tokens for start and end of word. Print the bigrams for the first three words.

**Test case:**

```console
python task03.py
```

```console
<S> e
e m
m m
m a
a <E>
<S> o
o l
l i
i v
v i
i a
a <E>
<S> a
a v
v a
a <E>
```

## Task 4

**Description:**

Count the bigrams obtained for all words and sort them in decreasing order of the number of times they occur in the dataset. Output the `15` entries that occur most often.

**Test case:**

```console
python task04.py
```

```console
[(('n', '<E>'), 6763), (('a', '<E>'), 6640), (('a', 'n'), 5438), (('<S>', 'a'), 4410), (('e', '<E>'), 3983), (('a', 'r'), 3264), (('e', 'l'), 3248), (('r', 'i'), 3033), (('n', 'a'), 2977), (('<S>', 'k'), 2963), (('l', 'e'), 2921), (('e', 'n'), 2675), (('l', 'a'), 2623), (('m', 'a'), 2590), (('<S>', 'm'), 2538)]
```

## Task 5

**Description:**

Store the counted bigrams in a 2D `PyTorch` tensor (matrix). The rows will represent the first character and the columns - the second character. Thus, each entry will tell us how often the second character follows the first. Output a heatmap-like figure showing the matrix.

**Test case:**

```console
python task05.py
```

![counts_01](../assets/w11_counts_01.png?raw=true "counts_01.png")

## Task 6

**Description:**

The figure from `Task 5` has unnecessary information - the bottom row is only zeros as is the second to last column. This is due to meaning:

- it's impossible for something to follow the `<E>` token;
- and it's impossible for the `<S>` token to follow a character.

Thus, we are wasting space enumerating impossible situations (bigrams). Instead of having two special tokens, let's replace `<S>` and `<E>` with `.` and place the enumerations of `.` to be the beginning of the matrix.

**Acceptance criteria:**

1. `<S>` and `<E>` are replaced with `.`.
2. The enumerations of `.` are in the beginning of the matrix.
3. The output below matches.

**Test case:**

```console
python task06.py
```

![counts_02](../assets/w11_counts_02.png?raw=true "counts_02.png")

## Task 7

**Description:**

Currently, we store the number of times a character follows another one. Each row describes the number of times a character is followed by every single character. This is good to have as a statistic, but it's not useful to a bigram language model. The goal of a bigram language model is to sample from a probability distribution every time it has to choose the next character. Thus, we have to normalize row-wise. Output the resulting probabilities.

Congratulations - you just created your very first bigram language model 🥳!

**Acceptance criteria:**

1. Every row in the output matrix sums up to `1`.

**Test case:**

```console
python task07.py
```

![probs_01](../assets/w11_probs_01.png?raw=true "probs_01.png")

## Task 8

**Description:**

Sample `50` words from the newly created character-level bigram language model. Use `seed=42`.

**Test case:**

```console
python task08.py
```

Approximate output:

```console
['ya', 'syahavilin', 'dleekahmangonya', 'tryahe', 'chen', 'ena', 'da', 'amiiae', 'a', 'keles', 'ly', 'a', 'oy', 'asityi', 'pepolannezale', 'shahlamion', 'nacelucyanarivieriaquten', 'kigshmole', 'ei', 'tonylyan', 'chaizelayane', 'ban', 'tann', 'jhkako', 'h', 'eyamilyi', 'zeilanaeleenoxtelyaylysara', 'nisema', 'malaae', 'karreeclilyai', 'bestlsa', 'ckennar', 'beriandeishliras', 'jalel', 'knai', 'kajaharon', 'da', 'mir', 'wzansuenayllea', 'i', 'marese', 'hiseeychararan', 'shnah', 'imincitompailiaanannn', 'lie', 'jossh', 'dizlyahath', 'ja', 'r', 'tsisararmaniand']
```

## Task 9

**Description:**

Calculate the loss of the model using `names.txt`.

Answer the following question in a comment: *Is the value of the loss better than what the most naive model would have?*.

**Acceptance criteria:**

1. The average negative log likelihood is used.
2. A comment is present with an answer to the question in the description.

**Test case:**

```console
python task09.py
```

```console
Loss: 2.4540144946949742
```

## Task 10

**Description:**

Calculate the probability that our model would output the word `simqeon`. Explain what you see.

**Acceptance criteria:**

1. A comment is written that explains why the output has the value that it has.

## Task 11

**Description:**

Fix the bug in the above task by adding `smoothing`.

**Acceptance criteria:**

1. `smoothing` is implemented and used during training.

**Test case:**

Now, if prompted to output the probability of predicting the word `simqeon`, the model should output something close to:

```console
Loss: 3.6458098739385605
```
