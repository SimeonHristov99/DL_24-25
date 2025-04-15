# 🎯 Goals for week 09

1. Practice defining, training, evaluating and improving recurrent neural networks.
2. Get started with Large Language Models.
3. Practice writing high quality code:
   1. Easy to read.
   2. Safe from bugs.
   3. Ready for change.

## Task 01

**Description:**

Let's tackle the problem of predicting electricity consumption based on past patterns. In our `DATA` folder you'll find the folder `electricity_consumption`. It contains electricity consumption in kilowatts, or kW, for a certain user recorded every 15 minutes for four years.

Define a function `create_sequences` that takes a `pandas` `Dataframe` and a sequence length and returns two `NumPy` arrays, one with input sequences and the other one with the corresponding targets. To test the correctness of the function, apply it on a dataset containing two columns with the numbers from `0` to `100` and output the first five entries in both arrays.

Apply the function on the training set of the electricity consumption data. Output the shape of the `X` and `y` splits and output the length of the `TensorDataset` for training the model.

**Acceptance criteria:**

1. A new function `create_sequences` is defined.
2. The first five entries after applying `create_sequences` on the `electricity_consumption` data are displayed.
3. An object of type `TensorDataset` is created and its length is outputted.

**Test case:**

```console
python task01.py
```

```console
First five training examples: [[0 1 2 3 4]
 [1 2 3 4 5]
 [2 3 4 5 6]
 [3 4 5 6 7]
 [4 5 6 7 8]]
First five target values: [5 6 7 8 9]

X_train.shape=(105119, 96)
y_train.shape=(105119,)
Length of training TensorDataset: 105,119
```

## Task 02

**Description:**

The model that we'll use to predict the electricity consumption would need to have a sequence-to-vector architecture.

Define a neural network that will have the following layers:

- two stacked `LSTM` layers with memory size of `32`. Initially, both types of memory will be all zeros;
- a linear layer that will take the recurrent layer's last output.

Apply the model on the data:

- train for `3` epochs with batches of `32` samples;
- use the mean squared error loss from the `torchmetrics` module;
- use the optimizer `AdamW` with a learning rate of `0.0001`.

Output the loss per epoch when training and the loss on the test set.

**Acceptance criteria:**

1. Two `LSTM` layers are stacked.
2. The correct shape for the linear layer is identified.
3. The mean squared error loss function from the `torchmetrics` module is used.
4. `tqdm` is used to visualize the progress through the batches.

**Test case:**

```console
python task02.py
```

```console
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3284/3284 [00:59<00:00, 55.45it/s]
Epoch 1, Average MSE loss: 0.19529657065868378
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3284/3284 [01:32<00:00, 35.69it/s]
Epoch 2, Average MSE loss: 0.12671414017677307
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3284/3284 [01:04<00:00, 50.63it/s]
Epoch 3, Average MSE loss: 0.1027834564447403
Test MSE: 0.09648480266332626
```

## Task 03

**Description:**

Try to create a better model than the one we obtained in the previous tasks. Treat this task as a playground for conducting experiments to understand what makes a good neural network.

**Acceptance criteria:**

1. A training process that results in a model giving lower MSE on the test set and a reasonable R-squared metric.

## Task 04

**Description:**

Run a summarization pipeline using the [`cnicu/t5-small-booksum`](https://huggingface.co/cnicu/t5-small-booksum) model from the Hugging Face hub.

A `long_text` about the Eiffel Tower has been provided below.

```python
long_text = "\nThe tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.\n"
```

Produce a summary of at most `50` tokens.

**Acceptance criteria:**

1. A summary of the given text is produced.

**Test case:**

```console
python task04.py
```

```console
the Eiffel Tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres
```

## Task 05

**Description:**

Pick an appropriate model from the [`huggingface hub`](https://huggingface.co/tasks) and use it to write a response to a customer review for the Riverview Hotel.

The review is provided below:

```python
review = "I had a wonderful stay at the Riverview Hotel! The staff were incredibly attentive and the amenities were top-notch. The only hiccup was a slight delay in room service, but that didn't overshadow the fantastic experience I had."
```

**Acceptance criteria:**

1. An appropriate LLM model is chosen.
2. Prompt engineering is used to guide the answer of the LLM.
3. The answer is coherent and logical.

**Test case:**

Due to different models you might pick, your outputs may be different.

```console
python task05.py
```

```console
Customer review:
I had a wonderful stay at the Riverview Hotel! The staff were incredibly attentive and the amenities were top-notch. The only hiccup was a slight delay in room service, but that didn't overshadow the fantastic experience I had.

Hotel reponse to the customer:
Dear valued customer, I am glad to hear you had a good stay with us. We appreciate your request, and look forward to our next visit next year. Thanks as always for the patience and feedback you offer us in your email!

Hotel reception:

Thank you,

Barbara J.
```

## Task 06

**Description:**

Pick an appropriate model from the [`huggingface hub`](https://huggingface.co/tasks) and use it to translate the following text from Spanish to Bulgarian.

```python
spanish_text = "Este curso sobre LLMs se está poniendo muy interesante"
```

**Acceptance criteria:**

1. An appropriate LLM model is chosen.

**Test case:**

Due to different models you might pick, your outputs may be different.

```console
python task06.py
```

```console
Този курс за LLMs става много интересен.
```

## Task 07

**Description:**

Let's using LLMs to answer the question `Who painted the Mona Lisa?` after giving them some text for context.

We've heard that are architecture differences in the different types of transformers. Let's see how they play out.

Pick two models from the [`huggingface hub`](https://huggingface.co/tasks):

- one for extractive question-answering;
- and another for generative question-answering.

Ask both of them the above question and output their answers.

```python
context = "The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci. Considered an archetypal masterpiece of the Italian Renaissance, it has been described as the most known, visited, talked about, and sung about work of art in the world. The painting's novel qualities include the subject's enigmatic expression, the monumentality of the composition, and the subtle modeling of forms."
```

**Acceptance criteria:**

1. Appropriate LLMs are chosen.

**Test case:**

Due to different models you might pick, your outputs may be different.

```console
python task07.py
```

```console
Answer extractive question-answering: Leonardo da Vinci
Answer generative question-answering:  is a half-length portrait painting by Italian artist Leonardo da
```

## Task 08

**Description:**

You want to leverage a pre-trained model from Hugging Face and fine-tune it with data from your company support team to help classify interactions depending on the risk for churn. This will help the team prioritize what to address first, and how to address it, making them more proactive. Customers classified as `1` are high risk, while those classified as `0` are low risk.

Prepare the training **and** test data for fine-tuning by tokenizing the text. With the sole purpose of practicing what we learned in class, we'll do the tokenization a bit unconventionally:

- Tokenize the training set by directly calling the tokenizer, so as to get a dictionary.
- Tokenize the testing set by row, so as to get a `Dataset` class.

We can load the data into a HuggingFace `Dataset` class using the method [`from_dict`](https://huggingface.co/docs/datasets/v3.2.0/package_reference/main_classes#datasets.Dataset.from_dict).

```python
train_data = {'text': ["I'm really upset with the delays on delivering this item. Where is it?",
"The support I've had on this issue has been terrible and really unhelpful. Why will no one help me?",
'I have a question about how to use this product. Can you help me?',
'This product is listed as out of stock. When will it be available again?'],
'label': [1, 1, 0, 0]}

test_data = {'text': ['You charged me twice for the one item. I need a refund.', 'Very good - thank you!'],
'label': [1, 0]}
```

Choose appropriate values for the hyperparameters `return_tensors`, `padding`, `truncation` and `max_length`.

**Acceptance criteria:**

1. A Python dictionary object is turned into a HuggingFace `Dataset` object.
2. Appropriate pre-trained tokenizer is used.
3. The resulting dataset has `4` columns named `text`, `label`, `input_ids` and `attention_mask`.

**Test case:**

Due to different pre-trained tokenizers you might pick, your outputs may be different.

```console
python task08.py
```

```console
Tokenized training data:
{'input_ids': tensor([[  101,  1045,  1005,  1049,  2428,  6314,  2007,  1996, 14350,  2006,
         12771,  2023,  8875,  1012,  2073,  2003,  2009,  1029,   102,     0],
        [  101,  1996,  2490,  1045,  1005,  2310,  2018,  2006,  2023,  3277,
          2038,  2042,  6659,  1998,  2428,  4895, 16001, 14376,  5313,   102],
        [  101,  1045,  2031,  1037,  3160,  2055,  2129,  2000,  2224,  2023,
          4031,  1012,  2064,  2017,  2393,  2033,  1029,   102,     0,     0],
        [  101,  2023,  4031,  2003,  3205,  2004,  2041,  1997,  4518,  1012,
          2043,  2097,  2009,  2022,  2800,  2153,  1029,   102,     0,     0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])}
Tokenized test data:
Dataset({
    features: ['text', 'label', 'input_ids', 'attention_mask'],
    num_rows: 1
})
```

## Task 09

**Description:**

Fine tune a model from the HuggingFace hub to perform binary risk identification using the above tokenized data.

After use it on the below samples:

```python
input_text = ["I'd just like to say, I love the product! Thank you!", "Worst product I've used in my entire life!", "Disgusting..."]
```

As a last step save the fine-tuned model and tokenizer in a new directory called `finetuned-customer-support-model`.

**Acceptance criteria:**

1. Fine-tuning is performed using the classes `Trainer` and `TrainingArguments`.
2. Fine-tuned model is applied to new data.
3. Fine-tuned model is saved to disk.

## Task 10

**Description:**

Let's set-up a one-shot learning example for a model that classifies sentiment of a text.

Here is the initial prompt:

```python
"""
Text: "The dinner we had was great and the service too."
Classify the sentiment of this sentence as either positive or negative.
"""
```

Your task is to turn this prompt into one that can be used for `2`-shot sentiment classification.

After you construct the new prompt, pass it to a model and output the result.

**Acceptance criteria:**

1. An appropriate model for sentiment classification is used.
2. The given prompt is extended so that `2`-shot classification is performed.

**Test case:**

```console
python task10.py
```

```console
POSITIVE
```

## Task 11

**Description:**

Below you'll see a sample text that is the start of a sentence. Use a pre-trained LLM of your choice from the HuggingFace hub to generate the rest of the sentence. Output the generated text and the mean perplexity of the model.

```python
prompt = 'Current trends show that by 2030'
```

What does this value mean?

**Acceptance criteria:**

1. An appropriate pre-trained model is used.
2. The mean perplexity is outputted to the console.
3. An answer to the question in the description is present as a comment.

**Test case:**

```console
python task11.py
```

```console
Generated Text: Current trends show that by 2030, the number of people living in poverty will be at its lowest level
Mean perplexity: 9.096508979797363
```

## Task 12

**Description:**

Let's evaluate the quality of a Spanish-to-English translation model.

Pick and appropriate model from the HuggingFace hub, translate the below given sentences and output the value of the most appropriate metric to use for this task.

```python
input_sentence_1 = "Hola, ¿cómo estás?"

reference_1 = [
     ["Hello, how are you?", "Hi, how are you?"]
     ]

input_sentences_2 = ["Hola, ¿cómo estás?", "Estoy genial, gracias."]

references_2 = [
     ["Hello, how are you?", "Hi, how are you?"],
     ["I'm great, thanks.", "I'm great, thank you."]
     ]
```

**Acceptance criteria:**

1. An appropriate pre-trained model is used.
2. All translations are outputted.
3. The appropriate metric's values.

**Test case:**

```console
python task12.py
```

```console
Translation for "input_sentence_1": Hey, how are you?
Metric for "input_sentence_1": <fill-in>
Translations for "input_sentence_2": ['Hey, how are you?', "I'm great, thanks."]
Metric for "input_sentence_2": <fill-in>
```

## Task 13

**Description:**

Let's compute metrics based on different scenarios.

*Scenario 1:*

You have been provided with a model-generated summary and a reference summary for validation. Use an appropriate metric to see how well the model performed.

```python
predictions = ["""Pluto is a dwarf planet in our solar system, located in the Kuiper Belt beyond Neptune, and was formerly considered the ninth planet until its reclassification in 2006."""]
references = ["""Pluto is a dwarf planet in the solar system, located in the Kuiper Belt beyond Neptune, and was previously deemed as a planet until it was reclassified in 2006."""]
```

*Scenario 2:*

Compare the results of machine translation and a reference text.

```python
generated = ["The burrow stretched forward like a narrow corridor for a while, then plunged abruptly downward, so quickly that Alice had no chance to stop herself before she was tumbling into an extremely deep shaft."]
reference = ["The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well."]
```

*Scenario 3:*

Evaluate the performance of an extractive QA model.

```python
predictions = ["It's a wonderful day", "I love dogs", "DataCamp has great AI courses", "Sunshine and flowers"]
references = ["What a wonderful day", "I love cats", "DataCamp has great AI courses", "Sunsets and flowers"]
```

**Acceptance criteria:**

1. The right metrics are chosen and outputted for each of the scenarios.
