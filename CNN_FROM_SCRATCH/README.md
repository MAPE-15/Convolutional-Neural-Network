# Convolutional Neural Network From Sctrach
Do not expect this neural network to be as precise as from the frameworks.
This neural network is programmed from scratch, to understand how it works.
This neural network is trained to recognize facial expressions and to classify each.  
That's why we use convolutional neural network.


## Facial Expressions:
- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise


## Data
Data is saved in the ./data/ directory.  
Data is separated into training set and validation - testing set.  
Training set is in the ./data/test/ directory.  
Validation - testing set is in the ./data/test/ directory

Data consists of images 48 x 48 in greyscale. In each image is a person, more specifically, person's face. This person's face has a specific facial expression.  


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
(m + 2p  x  n + 2p  x  c) * (f x f x c) = ( (m + 2p - f) / s + 1) x ( (n + 2p - f) / s + 1) x c
```

