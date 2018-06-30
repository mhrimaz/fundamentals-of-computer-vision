
# Adding Gausian Blur to an Image


```python
# Introduction to Computer Vision (Undergrad)
# School of Computer Engineering
# K.N. Toosi University of Technology

import numpy as np
import cv2

#by default image is rgb
I = cv2.imread('isfahan.jpg', cv2.IMREAD_GRAYSCALE);

# convert I to floating point from unsigned integer
# Note: For displaying floating point images the maximum
# intensity has to be 1 instead of 255
# convert to floating point and put into the range of [0,1]
I = I.astype(np.float) / 255


# create the noise image
sigma = 10 # notice maximum intensity is 1
N = np.random.randn(*I.shape) * sigma
# * opertator unpack array


# add noise to the original image
J = I+N; # or use cv2.add(I,N);

cv2.imshow('original',I)
cv2.waitKey(0) # press any key to exit

cv2.imshow('noisy image',J)
cv2.waitKey(0) # press any key to exit

cv2.destroyAllWindows()

```


```python
# random gausian with mean 0 and variance 1
np.random.randn(2, 4 ) * 0.01

```




    array([[-0.0069093 , -0.00249939,  0.02044273,  0.00092371],
           [ 0.00427555,  0.00239688,  0.01748823,  0.00469716]])



# Task 1


```python
import numpy as np
import cv2

I = cv2.imread('isfahan.jpg', cv2.IMREAD_GRAYSCALE);
I = I.astype(np.float) / 255

sigma = 0.04 # initial standard deviation of noise 

while True:
    N = np.random.randn(*I.shape) * sigma
    # * opertator unpack array


    # add noise to the original image
    J = I+N; # or use cv2.add(I,N);
    
    cv2.imshow('snow noise',J)
    
    # press any key to exit
    key = cv2.waitKey(33)
    if key & 0xFF == ord('u'): # if 'u' is pressed 
        sigma+= 0.01
        pass # increase noise
    elif key & 0xFF == ord('d'):  # if 'd' is pressed 
        if sigma >= 0:
            sigma-= 0.01
        pass # decrease noise 
    elif key & 0xFF == ord('q'):  # if 'q' is pressed then 
        break # quit
    
cv2.destroyAllWindows()
```

# Image Smoothing/Bluring


```python
import numpy as np
import cv2

I = cv2.imread('isfahan.jpg').astype(np.float64) / 255;

# display the original image
cv2.imshow('original',I)
cv2.waitKey()


# creating a box filter
m = 1 # choose filter size

while m<50:
    # create an m by m box filter
    # division is for filter adjuctment because 
    # the sum should be one so we are literally averaging
    F = np.ones((m,m), np.float64)/(m*m)
    #print F
    
    # Now, filter the image
    J = cv2.filter2D(I,-1, F)
    cv2.imshow('blurred',J)
    cv2.waitKey()
    m+=2

cv2.destroyAllWindows()
```


```python
np.ones((4,4), np.float64)
```




    array([[ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.]])




```python
np.ones((4,4), np.float64) / (4*4)
```




    array([[ 0.0625,  0.0625,  0.0625,  0.0625],
           [ 0.0625,  0.0625,  0.0625,  0.0625],
           [ 0.0625,  0.0625,  0.0625,  0.0625],
           [ 0.0625,  0.0625,  0.0625,  0.0625]])




```python
np.sum(np.ones((4,4), np.float64) / (4*4))
```




    1.0



# Smoothing with Gaussian Kernel


```python
import numpy as np
import cv2

I = cv2.imread('isfahan.jpg').astype(np.float64) / 255;

m = 7; # we will create an m by m filter

# create a 1D Gaussian filter
Fg = cv2.getGaussianKernel(m, sigma=-1);
# by setting sigma=-1, the value of sigma is computed 
# automatically as: sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8

print 'Before Dot: ',Fg
print Fg.shape # Fg is 1-dimensional (m by 1)



# Now we create a 2D filter
# We use matrix multiplication to create an m by m 2D filter
# out of "m by 1" and "1 by m" 1D filters, which in this case happens
# to be the same thing as correlation between 1D filters
Fg =  Fg.dot(Fg.T) # an "m by 1" matrix multiplied by a "1 by m" matrix

print 'After Dot: ',Fg
print Fg.shape



# filter the image with the Gaussian filter
Jg = cv2.filter2D(I,-1, Fg)

cv2.imshow('original',I)
cv2.waitKey()

cv2.imshow('blurred_Gaussian',Jg)
cv2.waitKey()

cv2.destroyAllWindows()
```

    Before Dot:  [[ 0.03125 ]
     [ 0.109375]
     [ 0.21875 ]
     [ 0.28125 ]
     [ 0.21875 ]
     [ 0.109375]
     [ 0.03125 ]]
    (7, 1)
    After Dot:  [[ 0.00097656  0.00341797  0.00683594  0.00878906  0.00683594  0.00341797
       0.00097656]
     [ 0.00341797  0.01196289  0.02392578  0.03076172  0.02392578  0.01196289
       0.00341797]
     [ 0.00683594  0.02392578  0.04785156  0.06152344  0.04785156  0.02392578
       0.00683594]
     [ 0.00878906  0.03076172  0.06152344  0.07910156  0.06152344  0.03076172
       0.00878906]
     [ 0.00683594  0.02392578  0.04785156  0.06152344  0.04785156  0.02392578
       0.00683594]
     [ 0.00341797  0.01196289  0.02392578  0.03076172  0.02392578  0.01196289
       0.00341797]
     [ 0.00097656  0.00341797  0.00683594  0.00878906  0.00683594  0.00341797
       0.00097656]]
    (7, 7)


# Task 2


```python
import numpy as np
import cv2

I = cv2.imread('isfahan.jpg').astype(np.float64) / 255;

noise_sigma = 0.04 # initial standard deviation of noise

m = 1; # initial filter size,
# with m = 1 the input image will not change

filter = 'b' # box filter

while True:
   
    if filter == 'b':
        # filter with a box filter
        F = np.ones((m,m), np.float64)/(m*m)
        
    elif filter == 'g':  
        F = cv2.getGaussianKernel(m, sigma=-1);
        F =  F.dot(F.T)
        # filter with a Gaussian filter
        pass
    
    # add noise to image
    N = np.random.randn(*I.shape) * noise_sigma
    J = I + N;
    

    # filtered image
    K = cv2.filter2D(J, -1, F);
    

    cv2.imshow('img', K)
    key = cv2.waitKey(30) & 0xFF
        

    if key == ord('b'):
        filter = 'b' # box filter
        print 'Box filter'
        
    elif key == ord('g'):  
        filter = 'g' # filter with a Gaussian filter
        print 'Gaussian filter'

    elif key == ord('+'):
        # increase m
        m = m + 2
        print 'm= ',m

    elif key == ord('-'):
        # decrease m
        if m >= 3:
            m = m - 2
        print 'm= ', m

    elif key == ord('u'):
        noise_sigma+= 0.01
        
    elif key  == ord('d'):  # if 'd' is pressed 
        if noise_sigma >= 0:
            noise_sigma-= 0.01
        

    elif key == ord('q'): 
        break # quit

cv2.destroyAllWindows()
```
