# Introduction to Neural Networks

---

## Basic terminology

---

---

## Data Section

---

## Data Representation

- Scalars
  - A scalar is just a single number.
- Vectors
  - A vector is an array of numbers.
  - The numbers are arranged in order.
  - We can identify each individual number by its index in that ordering.
  - We can think of vectors as identifying points in space, with each element giving the coordinate along a different axis.
- Matrices
  - A matrix is a 2-D array of numbers.
  - Each element is identified by two indices instead of just one.
  - Vectors can be thought of as matrices that contain only one column.
  - A scalar can be thought of as a matrix with only a single entry.
- Tensors
  - Tensor is an array with more than two axes.

## Data splits

- Training set.
- Testing set.
- Validation set.

---

## Algorithms Section

---

## Learning Algorithms

    A machine learning algorithm is an algorithm that is able to learn from data.

### Task

1. Classification
    - In this type of task, the computer program is asked to specify which of k categories some input belongs to.
    - Output is descrite.
    - For example, decide if the image contains a cat or a dog.
    - If  `y = f(x)`
        - Y is numeric value represents the calss.
        - X is the input to be classified.

2. Regression
    - In this type of task, the computer program is asked to predict a numerical value given some input.
    - Output is continuous value.
    - For example, predict house price based on location and size.

3. Transcription
    - In this type of task, the machine learning system is asked to observe a relatively unstructured representation of some kind of data and transcribe the information into discrete textual form.
    - Output is textual form.
    - For example, in optical character recognition, the computer program is shown a photograph containing an image of text and is asked to return this text in the form of a sequence of characters

4. Machine translation
    - In a machine translation task, the input already consists of a sequence of symbols in some language, and the computer program must convert this into a sequence of symbols in another language.
    - Input and Output are in textual format.

5. Structured output
    - Structured output tasks involve any task where the output is a vector (or other data structure containing multiple values) with important relationships between the different elements.
    - This is a generalization for `3 and 4`

6. Anomaly detection
    - In this type of task, the computer program sifts through a set of events or objects and flags some of them as being unusual or atypical.
    - An example of an anomaly detection task is credit card fraud detection. By modeling your purchasing habits, a credit card company can detect misuse of your cards.

7. Synthesis and sampling
    - In this type of task, the machine learning algorithm is asked to generate new examples that are similar to those in the training data.
    - For example in Video games, generating textures for objects and landscapes rather than asking an artist to do this.

8. Imputation of missing values
    - In this type of task, the machine learning algorithm is given a new example with some entries missing.
    - The algorithm must provide a prediction of the values of the missing entries.

9. Denoising
    - In this type of task, the machine learning algorithm is given in input a corrupted example and the algorithm must produce a clean output.

---

## Model Experience Section

---

- Supervised
  - The model experiences a dataset containing features, but each example is also associated with a label or target.
  - The algorithm “learns” from the training dataset by iteratively making predictions on the data and adjusting for the correct answer.
  - Tend to be more accurate than unsupervised learning models.
  - Require upfront human intervention to label the data appropriately.

- UnSupervised
  - The model experiences a dataset containing many features, then learn useful properties of the structure of this dataset.
  - Work on its own to discover the inherent structure of unlabeled data.
  - Require some human intervention for validating output variables.
  - For example, an unsupervised learning model can identify that online shoppers often purchase groups of products at the same time.
  However, a data analyst would need to validate that it makes sense for a recommendation engine to group baby clothes with an order of diapers, applesauce and sippy cups.

- Semi-supervised learning
  - Semi-supervised learning is a happy medium, where you use a training dataset with both labeled and unlabeled data.
  - Useful when it’s difficult to extract relevant features from data — and when you have a high volume of data.
  - Ideal for medical images, where a small amount of training data can lead to a significant improvement in accuracy.
  - For example, a radiologist can label a small subset of CT scans for tumors or diseases so the machine can more accurately predict which patients might require more medical attention.

---

## Performance Section

---

## Capcity

    The central challenge in machine learning is that our algorithm must perform well on new, previously unseeninputs not just those on which our model was trained.
    The ability to perform well on previously unobserved inputs is called **generalization**.

- Training error.
  - How the model performed on the trainig data.
- Generalization error (test error).
  - How the model performed on the test sample.
- Underfitting.
  - Both test and training errors are bad.
  - **What do you think is the problem?**
- Overfitting.
  - Training error is small but testing error is high.

![Image](Files/Intro/fitting01.png)

So, always you need to the following:

1. Make the training error small.
2. Make the gap between training and test error small.

![Image](Files/Intro/fitting02.png)

---

## Neural Networks Section

---

## Neurons

- Neuron is tha basic building block of the neural network.
- It is derived from the brain neurons.

![Image](Files/Intro/neuron01.png)

1. receive information from many other neurons,
2. aggregate this information via changes in cell voltage at the cell body, and
3. transmit a signal if the cell voltage crosses a threshold level, which can be received by

![Image](Files/Intro/neuron02.png)

1. receive input from multiple other neurons,
2. aggregate those inputs via a simple arithmetic operation called the weighted sum,
and
3. generate an output if this weighted sum crosses a threshold level, which can then be sent on to many other neurons within a network.

Neuron output:

![Image](Files/Intro/neuron03.png)

## Activation Function

Neuron output as we saw in the previous image is binary.
This limits the learning operation and narrows range of possible inputs and outputs.
Hence, Applying an activation function to introduce non-linearity.

1. Sigmoid
Applies the following function
![Image](Files/Intro/activation01.png)<br>
![Image](Files/Intro/activation02.png)

2. Tanh
Applies the following function
![Image](Files/Intro/activation03.png)<br>
![Image](Files/Intro/activation04.png)

3. RELU
Applies the following function `a = max(0, z)`<br>
![Image](Files/Intro/activation05.png)

## Resourses

- [Keras Official Guide](https://keras.io/guides/)

- Books
  - [The deep learning book](https://www.deeplearningbook.org)
  - [Neural networks and deep learning](http://neuralnetworksanddeeplearning.com/)

- Play around
  - [TF Playground](https://playground.tensorflow.org)
  - [pix-to-pix](https://affinelayer.com/pixsrv/index.html)

- Videos
  - [3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
