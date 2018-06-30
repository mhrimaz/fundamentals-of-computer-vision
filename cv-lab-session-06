
## Bean Counting


```python
import numpy as np
import cv2

I = cv2.imread('beans.jpg')
G = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)

ret, T = cv2.threshold(G,127,255,cv2.THRESH_BINARY)

cv2.imshow('Thresholded', T)
cv2.waitKey(0) # press any key to continue...

## erosion 
kernel = np.ones((19,19),np.uint8)
T = cv2.erode(T,kernel)
cv2.imshow('After Erosion', T)
cv2.waitKey(0) # press any key to continue...

n,C = cv2.connectedComponents(T);

font = cv2.FONT_HERSHEY_SIMPLEX 
cv2.putText(T,'There are %d beans!'%(n-1),(20,40), font, 1, 255,2)
cv2.imshow('Num', T)
cv2.waitKey(0)

cv2.destroyAllWindows()
```

## Simple Background Substraction


```python
import numpy as np
import cv2

I1 = cv2.imread('scene1.jpg')
I2 = cv2.imread('scene2.jpg')

cv2.imshow('Image 1 (background)', I1)
cv2.waitKey(0)

cv2.imshow('Image 2', I2)
cv2.waitKey(0)

K = np.abs(np.int16(I2)-np.int16(I1)) # take the (signed int) differnce
K = K.max(axis=2) # choose the maximum value over color channels
K = np.uint8(K)
cv2.imshow('The difference image', K)
cv2.waitKey(0)

threshold = 45
ret, T = cv2.threshold(K,threshold,255,cv2.THRESH_BINARY)
cv2.imshow('Thresholded', T)
cv2.waitKey(0)

## opening
kernel = np.ones((3,3),np.uint8)
T = cv2.morphologyEx(T, cv2.MORPH_OPEN, kernel)
cv2.imshow('After Openning', T)
cv2.waitKey(0)

## closing
kernel = np.ones((22,22),np.uint8)
T = cv2.morphologyEx(T, cv2.MORPH_CLOSE, kernel)
cv2.imshow('After Closing', T)
cv2.waitKey(0)

n,C = cv2.connectedComponents(T);

J = I2.copy()
J[T != 0] = [255,255,255]
font = cv2.FONT_HERSHEY_SIMPLEX 
cv2.putText(J,'There are %d toys!'%(n-1),(20,40), font, 1,(0,0,255),2)
cv2.imshow('Number', J)
cv2.waitKey()
  
## connected components with statistics
n,C,stats, centroids = cv2.connectedComponentsWithStats(T);

j = n-1
maxArea = -1

for i in range(n):
    print "-"*20
    print "Connected Component: ", i
    print "center= %.2f,%.2f"%(centroids[i][0], centroids[i][1])
    print "left= ", stats[i][0]
    print "top=  ",  stats[i][1]
    print "width=  ", stats[i][2]
    print "height= ", stats[i][3]
    print "area= ", stats[i][4]
    if(stats[i][4]>maxArea and (i is not 0)):
        maxArea = stats[i][4]
        j = i
    

 # j: index of largest connected component (change this line)
J[C == j] = [0,0,255] # Paint largest connected component in RED
cv2.imshow('Largest Toy in red', J)
cv2.waitKey()

cv2.destroyAllWindows()

```

    --------------------
    Connected Component:  0
    center= 410.79,236.48
    left=  0
    top=   0
    width=   816
    height=  459
    area=  326686
    --------------------
    Connected Component:  1
    center= 402.21,154.15
    left=  324
    top=   34
    width=   167
    height=  215
    area=  21324
    --------------------
    Connected Component:  2
    center= 239.21,184.26
    left=  163
    top=   81
    width=   162
    height=  173
    area=  12986
    --------------------
    Connected Component:  3
    center= 717.84,182.54
    left=  676
    top=   142
    width=   86
    height=  82
    area=  6213
    --------------------
    Connected Component:  4
    center= 73.92,216.39
    left=  35
    top=   180
    width=   78
    height=  72
    area=  4067
    --------------------
    Connected Component:  5
    center= 572.91,201.07
    left=  554
    top=   181
    width=   39
    height=  39
    area=  1261
    --------------------
    Connected Component:  6
    center= 628.56,283.17
    left=  607
    top=   250
    width=   47
    height=  60
    area=  2007
