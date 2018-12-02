# Clothes_Classification_with-Keras

the project is on clothes classification .i used mnist dataset(fashion_mnist) for the classification application 
The images each are 28 x 28 arrays, with pixel values ranging between 0 and 255

 The labels are arrays of integers, ranging from 0 to 9. These correspond to the class of clothing the image represents:

Digit	Class

0	T-shirt/top

1	Trouser

2	Pullover

3	Dress

4	Coat

5	Sandal

6	Shirt

7	Sneaker

8	Bag

9	Ankle boot

The core data structure of Keras is a model, a way to organize layers. The simplest type of model is the Sequential model, a linear stack of layers.

 the network consists of a sequence of two dense layers

The first dense layer has 128 nodes (or neurons) using a relu activation function 

 the rectifier is an activation function defined as the positive part of its argument:
 f(x)=x^{+}=\max(0,x)
 A unit employing the rectifier is also called a rectified linear unit (ReLU).
 
The second (and last) layer is a 10-node using softmax activation function 

 the softmax function, softargmax,[1] or normalized exponential function, is a generalization of the logistic function that "squashes" a K-dimensional vector Z of arbitrary real values to a K-dimensional vector sigma(z) of real values, for K>=2, where each entry is in the interval (0, 1), and all the entries add up to 1.


 —this returns an array of 10 probability scores that sum to 1. Each node contains a score that indicates the probability that the current image belongs to one of the 10 digit classes.

