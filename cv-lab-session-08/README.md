
## Corner Detection


```python
import cv2
import numpy as np

I = cv2.imread('square.jpg')
G = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)

G = np.float32(G)
window_size = 2
soble_kernel_size  = 3 # kernel size for gradients
alpha = 0.04
H = cv2.cornerHarris(G,window_size,soble_kernel_size,alpha)

# normalize C so that the maximum value is 1
H = H / H.max()

# C[i,j] == 255 if H[i,j] > 0.01, and C[i,j] == 0 otherwise
C = np.uint8(H > 0.005) * 255

## connected components
nc,CC = cv2.connectedComponents(C);

CC = np.uint8(CC)
# to count the number of corners we count the number
# of nonzero elements of C
n = np.count_nonzero(CC)
CC[CC != 0]=255
# Show corners as red pixels in the original image
I[CC != 0] = [0,0,255]


cv2.imshow('corners',CC)
cv2.waitKey(0) # press any key

font = cv2.FONT_HERSHEY_SIMPLEX 
cv2.putText(I,'There are %d corners!'%(nc-1),(20,40), font, 1,(0,0,255),2)
cv2.imshow('corners',I)
cv2.waitKey(0) # press any key

cv2.destroyAllWindows()

```


```python
import cv2
import numpy as np
import glob

fnames = glob.glob('*.jpg')

for filename in fnames:
    
    I = cv2.imread(filename)
    G = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
    G = np.float32(G)

    window_size = 3
    soble_kernel_size  = 3 # kernel size for gradients
    alpha = 0.04
    H = cv2.cornerHarris(G,window_size,soble_kernel_size,alpha)
    H = H / H.max()

    C = np.uint8(H > 0.01) * 255
    J = I.copy()
    J[C != 0] = [0,0,255]
    cv2.imshow('corners',J)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break


    # plot centroids of connected components as corner locations
    nC, CC, stats, centroids = cv2.connectedComponentsWithStats(C)

    J = I.copy()
    for i in range(1,nC):
        cv2.circle(J, (int(centroids[i,0]), int(centroids[i,1])), 3, (0,0,255))
    cv2.imshow('corners',J)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

        

    # fine-tune corner locations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(G,np.float32(centroids),(5,5),(-1,-1),criteria)
    J = I.copy()
    for i in range(1,nC):
        cv2.circle(J, (int(corners[i,0]), int(corners[i,1])), 3, (0,0,255))
    cv2.imshow('corners',J)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()

```

## TASK 1


```python
import cv2
import numpy as np


I = cv2.imread('polygons.jpg')
G = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)

ret, T = cv2.threshold(G,220,255,cv2.THRESH_BINARY_INV)
nc,CC = cv2.connectedComponents(T)
window_size = 5
Sobel_kernel_size = 3
alpha = 0.04
A = np.float32
for k in range(1,nc):

    Ck = np.zeros(T.shape, dtype=np.float32)
    Ck[CC == k] = 1;
    Ck = cv2.GaussianBlur(Ck,(5,5),0)

    H = cv2.cornerHarris(Ck, window_size, Sobel_kernel_size, alpha)
    H = H / H.max()

    C = np.uint8(H > 0.01) * 255

    nc, con = cv2.connectedComponents(C);



    Ck = cv2.cvtColor(Ck,cv2.COLOR_GRAY2BGR)

    I[con!=0] = [0,0,255]
    Ck[con!=0] = [0,0,255]

    cv2.imshow('han',I)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(Ck,'There are %d vertices!'%(nc-1),(20,30), font, 1,(0,0,255),1)

    
    cv2.imshow('corners',Ck)
    cv2.waitKey(0) # press any key


        
cv2.destroyAllWindows()
```
