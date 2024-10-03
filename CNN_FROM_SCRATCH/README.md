# Convolutional Neural Network From Scratch
Do not expect this neural network to be as precise as from the frameworks.
This neural network is programmed from scratch, to understand how it works.
This neural network is trained to recognize facial expressions and to classify each.
We train the model to predict a facial expression from person's face in a static image.  
A real live prediction of facial expressions via webcam is in the CNN where actual frameworks are used.  

## Table of Contents
1. [Convolutional Neural Network From Scratch](#convolutional-neural-network-from-scratch)
2. [Table of Contents](#table-of-contents)
3. [Facial Expressions](#facial-expressions)
4. [Data](#data)
5. [Convolutional Layers](#convolutional-layers)
   - [Convolutional Operations](#convolutional-operations)
   - [Padding](#padding)
   - [Stride](#stride)

## Facial Expressions
- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise


## Data
Data is saved in the ./data_rgb/ or ./data_greyscale/ directory.  
Data is separated into training set and validation - testing set.  
Training set is in the ./train/ directory.  
Validation - testing set is in the ./test/ directory.  
./data_rgb/ provides RGB images of higher resolution, but it has fewer samples than ./data_greyscale/.  
./data_greyscale/ provides greyscale images of 48 x 48 resolution, has much more samples than ./data_rgb/.  

Data consists of images. In each image is a person, more specifically, person's face. This person's face has a specific facial expression.  


## Training
To train the model, first for each facial expression a bunch of images are read and converted to normalized numpy arrays adn their labels.
Label names (like "happy", "sad", etc...) are converted into labels, these labels are one-hot encoded vector representations of the label names.
The input into the model will be the image numpy arrays, the output is a one-hot vector representing a predicted facial expression.

## Convolutional Layers
Convolution layers consists of multiple filters / kernels.  
Each filter is one convolution layer.

This filter can be of any size, let's say of size f x f. (f - width and height of the filter)  

We use <b>convolution operations</b> to the input image using the filter, also we can use techniques like <b>padding</b> to the input image, or we can specify a <b>stride</b> during convolution operation.

### Convolutional Operations

For example let's say we have a filter, this concrete filter is used to detect vertical edges.  
Size of this filter is 3x3, f = 3.

|   |   |    |
|---|---|----|
| 1 | 0 | -1 | 
| 1 | 0 | -1 |
| 1 | 0 | -1 |

And we have an input image:

|   |   |   |   |
|---|---|---|---|
| 2 | 3 | 8 | 3 |
| 1 | 3 | 9 | 1 |
| 1 | 5 | 6 | 1 |
| 5 | 0 | 0 | 1 |

Now that we have defined a filter, and we have an input image, let's apply a convolution operation to the input image using our filter. Convolution operation is input_image * filter. Multiplying input image with the filter.  

|   |   |   |   |
|---|---|---|---|
| 2 | 3 | 2 | 3 |
| 1 | 3 | 5 | 1 |   
| 4 | 5 | 1 | 1 |
| 5 | 0 | 0 | 1 |

- input image size: (4 x 4)

 |   |   |    |
 |---|---|----|
 | 1 | 0 | -1 | 
 | 1 | 0 | -1 |
 | 1 | 0 | -1 |
 
- filter size (3 x 3)


Result after multiplying would be:

|    |   |
|----|---|
| -2 | 6 |
| 4  | 5 |  

- output image size: (2 x 2)

```
Notice that the result image - ouput image (2 x 2) has much lower resolution than the original one - input image (4 x 4).
So the main reason we use convolutional operations is to reduce dimensionality, reduce the complexity of the image.
```

Formula to get the size of the output image after the convolution:

```
f = size fo kernel
m = width of input image
n = height of the input image

(m x n) * (f x f) = (m - f + 1) x (n - f + 1)
```

Image can have multiple channels - c. For RGB image there would be 3 channels, meaning 3 matrices. One matrix for RED, one for BLUE and one for GREEN.  
In the example above, there was only 1 channel. If input image is RGB, the input image matrix would be 3 dimensional, there would be 3 matrices.  
Then the filter must also have 3 channels, to be able to compute the output image matrix.  

For example lets say we have an RGB image:

|   |   |   |   |
|---|---|---|---|
| 2 | 3 | 2 | 3 |
| 1 | 3 | 5 | 1 |   
| 4 | 5 | 1 | 1 |
| 5 | 0 | 0 | 1 |

- For red

|   |   |   |   |
|---|---|---|---|
| 0 | 3 | 2 | 3 |
| 0 | 1 | 5 | 0 |   
| 4 | 5 | 3 | 1 |
| 1 | 0 | 0 | 4 |

- For green

|   |   |   |   |
|---|---|---|---|
| 2 | 0 | 1 | 4 |
| 1 | 0 | 5 | 2 |   
| 5 | 0 | 8 | 2 |
| 3 | 2 | 0 | 3 |

- For blue

In this example the image of size: (4 x 4 x 3), where 4 x 4 is the size of the image and 3 is the number of channels.

Then the filter shlould also have 3 channels:

 |   |   |    |
 |---|---|----|
 | 1 | 0 | -1 | 
 | 1 | 0 | -1 |
 | 1 | 0 | -1 |
 
- 1st channel

 |    |    |   |
 |----|----|---|
 | 0  | 0  | 1 | 
 | 0  | -1 | 1 |
 | -1 | 0  | 1 |
 
- 2nd channel 

 |   |   |   |
 |---|---|---|
 | 1 | 0 | 1 | 
 | 0 | 1 | 0 |
 | 1 | 0 | 1 |
 
- 3rd channel

Convolutional operation would be the same as in the example where there was only 1 channel.  
Filter is now of size (3 x 3 x 3).

```
Images can have a depth - channels, for example for RGB image there are 3 channels, for grayscale image there is only 1 channel, etc...
```

Formula to get the size of the output image after the convolution, now having number of channels - c:

```
c = number of channels
f = size fo kernel
m = width of input image
n = height of the input image

(m x n x c) * (f x f x c) = (m - f + 1) x (n - f + 1) x c

Notice that number of channels remain, only the output image size has been reduced.
```

```
As you can see, there is an infite number of filters that can be applied. We cannot hard code the filters and expect great performance from our model.
Therefore our model will train the filter values to get the best possible filters for our model.
```

### Padding

Now when you know, what convolutional operations look like and what they do.  
Simply put, the convolutional operation reduces the dimensionality of the image, reduces the resolution.  

As you seen in the example above, f.e. when applying a convolutional operation on the input image of size (4 x 4) with the filter - kernel of size (3 x 3) the resulting output image would be of size (2 x 2), so the size is greatly reduced.  
After applying many filters on the input image, the size will be significantly reduced, the size of the resulting image would be too small.  

The problem is that the output image can get so small, that some of the important information from the original image can be lost, meeaning the CNN would have a much harder time to learn.  

That's why we can use something called <b>padding</b>.

Let's say we have an input image of size (4 x 4):

|   |   |   |   |
|---|---|---|---|
| 2 | 3 | 2 | 3 |
| 1 | 3 | 5 | 1 |   
| 4 | 5 | 1 | 1 |
| 5 | 0 | 0 | 1 |

- input image size: (4 x 4)

If we apply padding `p = 1`, the input image would look like this:

|   |   |   |   |   |   |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 2 | 3 | 2 | 3 | 0 |
| 0 | 1 | 3 | 5 | 1 | 0 |
| 0 | 4 | 5 | 1 | 1 | 0 |
| 0 | 5 | 0 | 0 | 1 | 0 |
| 0 | 0 | 0 | 0 | 0 | 0 |

- input image of size (4 x 4) with padding p = 1, meaning the input image is now of size (6 x 6)

As you can see the padding wraps the image with zeros. If p = 2 it wraps the image 2 times, etc...

Now when applying a convolution operation on the input image (4 x 4) with padding = p = 1 (meaning the input image is of size (6 x 6)), with the filter (3 x 3), the result output image will have size (4 x 4).  
Output image size will be the same as the input image size is without the padding.

Let's see for ourselves: 

|   |   |   |   |   |   |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 2 | 3 | 2 | 3 | 0 |
| 0 | 1 | 3 | 5 | 1 | 0 |
| 0 | 4 | 5 | 1 | 1 | 0 |
| 0 | 5 | 0 | 0 | 1 | 0 |
| 0 | 0 | 0 | 0 | 0 | 0 |

- input image of size (4 x 4) with padding p = 1, meaning the input image is now of size (6 x 6)

 |   |   |    |
 |---|---|----|
 | 1 | 0 | -1 | 
 | 1 | 0 | -1 |
 | 1 | 0 | -1 |
 
- filter size (3 x 3)

Resuling output image after convolutional operation:

|     |    |   |   |
|-----|----|---|---|
| -6  | -4 | 2 | 7 |
| -11 | -1 | 6 | 8 |
| -8  | 4  | 5 | 6 |
| -5  | 8  | 3 | 1 |

- output image of size (4 x 4) - the same size as the input image without the padding

Formula to get the size of the output image after the convolution, having number of channels - c, and now having padding - p:

```
c = number of channels
p = padding applied to the input image
f = size fo kernel
m = width of input image
n = height of the input image

(m + 2p  x  n + 2p  x  c) * (f x f x c) = (m + 2p - f + 1) x (n + 2p - f + 1) x c
```

The size of the output image is `(m + 2p - f + 1) x (n + 2p - f + 1)`.

This size is the same size of the input image without the padding:  
`(m + 2p - f + 1) = m`  
`(n + 2p - f + 1) = n`  

We can now specify the padding p:  
```
p = (f - 1) / 2
```

When working with some frameworks, you can apply two types of convolutions - "VALID" or "SAME" Convolution.  
"VALID" convolution applies zero padding to the input image p = 0.  
"SAME" convolution applies the padding using the formula above `p = (f - 1) / 2`, where f is often an odd number.

### Stride

Stride is the shift of the filter window in the input image.  
Before now, we have been working with stride = s = 1, where filter was shifting in the input image by 1 pixel.

You can specify the shift - stride, filter - kernel window can move - shift more than 1 pixel.  

The purpose of defining a stride is to downsample images, retaining only essential information.  
This not only speeds up training and inference but also makes the models more manageable, especially when dealing with large images.

Let's say we again have an input image of size (4 x 4) and kernel of size (2 x 2), but now with stride = s = 2.

|   |   |   |   |
|---|---|---|---|
| 2 | 3 | 2 | 3 |
| 1 | 3 | 5 | 1 |   
| 4 | 5 | 1 | 1 |
| 5 | 0 | 0 | 1 |

- input image of size (4 x 4)

|   |   |
|---|---|
| 2 | 3 |
| 3 | 2 |

- filter of size (2 x 2)

Result of the convolutional operation with specified stride = s = 2:

|    |    |
|----|----|
| -8 | 8  |
| 8  | -3 |

- output image of size (2 x 2)

Formula to get the size of the output image after the convolution, having number of channels - c, and now having stride - s:

```
c = number of channels
p = stride applied during convolution operation
f = size fo kernel
m = width of input image
n = height of the input image

(m x n x c) * (f x f x c) = ( (m - f) / s + 1) x ( (n - f) / s + 1) x c
```

Formula using both padding - p and stride - s, where s != 0:
```
(m + 2p  x  n + 2p  x  c) * (f x f x c) = floor( ( (m + 2p - f) / s + 1) ) x floor( ( (n + 2p - f) / s + 1) ) x c
```


## Pooling Layers
There are many pooling layers, in this project we will use Max Pooling and Average Pooling layers.

Each polling layer constists of one pool.
This pool can be of any size, let's say of size p x p x c. (f - width and height of the pool, c - number of channels) this pool is EMPTY.  
We do not care about the pool values, we only care about the intersection of input image with this pool window.


### Max Pooling Layer
Reason to use a max pooling layer is simple.  
To reduce size of an image - reduce dimensionality, and keep most of the important features of the image intact.  

How does max pooling work?  
The empty pool shifts thorughout the whole image with specific stride.  
It takes the maximum value from the shifted window - the intersection, and saves it to the output.

Let's say we have an input image of size (4 x 4) and pool of size (2 x 2), with stride = s = 2.

|   |   |   |   |
|---|---|---|---|
| 2 | 3 | 2 | 3 |
| 1 | 3 | 5 | 1 |   
| 4 | 5 | 1 | 1 |
| 5 | 0 | 0 | 1 |

- input image of size (4 x 4)

|   |   |
|---|---|
| - | - |
| - | - |

- empty pool of size (2 x 2)

Result image after gone through max pooling layer.

|   |   |
|---|---|
| 3 | 5 |
| 5 | 1 |


Important notes
```
Use of max pooling layer is optional.
If you decide to use max pooling layer, always apply it after an image went through the convolutional layer. 
You do not need to put a max pooling layer after the convolutional layer, rule o thumb: there must be always more convolutional layers than max pooling layers.

Max pooling layer reduces dimensionality of the image -> training is faster -> model is faster.
We do not need any parameters in max pooling layer, therefore there is nothing to be trained in the max pooling layer.
```


### Average Pooling Layer
Reason to use an average pooling layer is also simple.  
To reduce size of an image - reduce dimensionality, and keep most of the important features of the image intact.  

The difference between max pooling layer is that the average pooling layer does not take a maximum value, but it takes an average value in the pooling window, which can help to retain more valuable information.  
The max pooling layer does not have that smoothing effect than the average pooling layer.

For some tasks where overall smooth features are important (e.g., facial recognition or medical image processing), average pooling may perform better than max pooling because it captures a broader representation of features.

How does average pooling work?  
The empty pool shifts thorughout the whole image with specific stride.  
It takes the average value from the values in the shifted window - the intersection, and saves it to the output.

Let's say we have an input image of size (4 x 4) and pool of size (2 x 2), with stride = s = 2.

|   |   |   |   |
|---|---|---|---|
| 2 | 3 | 2 | 3 |
| 1 | 3 | 5 | 1 |   
| 4 | 5 | 1 | 1 |
| 5 | 0 | 0 | 1 |

- input image of size (4 x 4)

|   |   |
|---|---|
| - | - |
| - | - |

- empty pool of size (2 x 2)

Result image after gone through average pooling layer.

|      |      |
|------|------|
| 2.25 | 2.75 |
| 3.5  | 0.75 |


Important notes
```
Use of average pooling layer is optional.
If you decide to use average pooling layer, always apply it after an image went through the convolutional layer. 
You do not need to put a average pooling layer after the convolutional layer, rule o thumb: there must be always more convolutional layers than average pooling layers.

Average pooling layer reduces dimensionality of the image -> training is faster -> model is faster.
We do not need any parameters in average pooling layer, therefore there is nothing to be trained in the average pooling layer.
```

### Max Pooling Layer vs Average Pooling Layer

What max pooling layer and average pooling layer have in common?
```
Both reduce the size of an image - reduce dimensionality which helps to lower computational cost and memory usage.
Both do not use any parameters and therefore they are non-trainable.
```

What is the difference between them?
```
Max pool takes the maximum value from the window, the average pool takes an average value between all values in the window.

Max pool retains the strongest features in the window, the average pool retains an average intensity of features - smooths the output.

Max pool is used to detect sharp, prominent features like edges or distinct features.
Average pool is used when you want to keep more subtle features or when no single features dominates the image.

Max pool often used in object detection or image classification.
Average pool often used in facial expression or texture analysis.
```