
## Robust estimation with RANSAC


```python
import numpy as np
import cv2

I1 = cv2.imread('obj3.jpg')
G1 = cv2.cvtColor(I1,cv2.COLOR_BGR2GRAY)

I2 = cv2.imread('scene.jpg')
G2 = cv2.cvtColor(I2,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create() # opencv 3
# use "sift = cv2.SIFT()" if the above fails

# detect keypoints and compute their disriptor vectors
keypoints1, desc1 = sift.detectAndCompute(G1, None); # opencv 3
keypoints2, desc2 = sift.detectAndCompute(G2, None); # opencv 3

print "No. of keypoints1 =", len(keypoints1)
print "No. of keypoints2 =", len(keypoints2)

print "Descriptors1.shape =", desc1.shape
print "Descriptors2.shape =", desc2.shape

# stop here!!
exit() # comment this line out to move on!

# brute-force matching
bf = cv2.BFMatcher()

# for each descriptor in desc1 find its
# two nearest neighbours in desc2
matches = bf.knnMatch(desc1,desc2, k=2)

good_matches = []
alpha = 0.75
for m1,m2 in matches:
    # m1 is the best match
    # m2 is the second best match
    if m1.distance < alpha *m2.distance:
        good_matches.append(m1)

# apply RANSAC
print keypoints1[0].pt
points1 = [keypoints1[m.queryIdx].pt for m in good_matches]
points1 = np.array(points1,dtype=np.float32)
print "points1: ",points1.shape

points2 = [keypoints2[m.trainIdx].pt for m in good_matches]
points2 = np.array(points2,dtype=np.float32)
print "points2: ",points2.shape
H, mask = cv2.findHomography(points1, points2, cv2.RANSAC,5.0) # 5 pixels margin
mask = mask.ravel().tolist()
print "mask.shape:",len(mask)
print "mask:",mask

good_matches = [m for m,msk in zip(good_matches,mask) if msk == 1]

I = cv2.drawMatches(I1,keypoints1,I2,keypoints2, good_matches, None)

cv2.imshow('sift_keypoints1',I)
cv2.waitKey()
cv2.destroyAllWindows()
```

    No. of keypoints1 = 539
    No. of keypoints2 = 3089
    Descriptors1.shape = (539, 128)
    Descriptors2.shape = (3089, 128)
    (4.232668399810791, 101.63789367675781)
    points1:  (162, 2)
    points2:  (162, 2)
    mask.shape: 162
    mask: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]


### Questions
- What are the shapes of points1 and points2 : points1->(162,2) , points2->(162,2)


- What is keypoints1[m.queryIdx].pt ? keypoints1 is key point in the first image, m is one element of good key point so we get the position of keypoint in the first image

- Print the variable mask. What does it represent? Good features, zero mean that the features is not good and it's an outlier

- What does the following line do?
good_matches ​ = [m ​for ​ m,msk ​in ​ ​zip ​(good_matches,mask) ​if ​ msk == 1]<br>
it store good matches if the correspondent mask value is equal to one

## TASK 1


```python
import numpy as np
import cv2
import glob

sift = cv2.xfeatures2d.SIFT_create() # opencv 3
# use "sift = cv2.SIFT()" if the above fails

I2 = cv2.imread('scene.jpg')
G2 = cv2.cvtColor(I2,cv2.COLOR_BGR2GRAY)
keypoints2, desc2 = sift.detectAndCompute(G2, None); # opencv 3

fnames = glob.glob('obj?.jpg')
fnames.sort()
for fname in fnames:

    I1 = cv2.imread(fname)
    G1 = cv2.cvtColor(I1,cv2.COLOR_BGR2GRAY)
    keypoints1, desc1 = sift.detectAndCompute(G1, None); # opencv 3

    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    alpha = 0.75
    for m1, m2 in matches:

        if m1.distance < alpha * m2.distance:
            good_matches.append(m1)

    points1 = [keypoints1[m.queryIdx].pt for m in good_matches]
    points1 = np.array(points1, dtype=np.float32)

    points2 = [keypoints2[m.trainIdx].pt for m in good_matches]
    points2 = np.array(points2, dtype=np.float32)
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)  # 5 pixels margin
    good_matches = [m for m, msk in zip(good_matches, mask) if msk == 1]

    print H
    J = cv2.warpPerspective(I1, H, (I2.shape[1],I2.shape[0]) )

    # alternatingly show images I2 and J
    ind = 0;
    imgs = [I2, J]
    while 1:
        ind = 1-ind

        cv2.imshow('Reg',imgs[ind])
        key =  cv2.waitKey(800) 
                
        if key & 0xFF == ord('q'):
            exit()
        elif key & 0xFF != 0xFF:
            break


        

cv2.destroyAllWindows()  
```

    [[ -2.59515195e-01  -7.09040248e-01   5.75526085e+02]
     [  6.00018409e-01  -2.85982197e-01   4.89590547e+02]
     [ -1.60029976e-04  -1.45996976e-04   1.00000000e+00]]
    [[ -3.89226709e-01  -6.59809569e-01   4.77500109e+02]
     [  5.75545370e-01  -4.28075774e-01   1.29012903e+02]
     [ -1.10193251e-04  -1.18691914e-04   1.00000000e+00]]
    [[ -2.15466149e-01  -3.26932233e-01   7.44818174e+02]
     [  2.90011005e-01  -1.63372447e-01   1.88784702e+02]
     [ -6.39607916e-05   3.96498953e-05   1.00000000e+00]]
    [[  2.41350732e-01  -1.26554774e-01   2.13121255e+02]
     [  1.04838961e-01   2.30653337e-01   4.34180951e+02]
     [ -1.62595520e-05  -3.42185332e-05   1.00000000e+00]]
    [[  7.94204572e-02  -3.45036979e-01   4.45331225e+02]
     [  3.27867637e-01   1.99268199e-02   2.39618528e+02]
     [  3.26810214e-05  -1.09749753e-04   1.00000000e+00]]


## TASK 2


```python
import numpy as np
import cv2
import glob

sift = cv2.xfeatures2d.SIFT_create() # opencv 3
# use "sift = cv2.SIFT()" if the above fails

I2 = cv2.imread('scene.jpg')
G2 = cv2.cvtColor(I2,cv2.COLOR_BGR2GRAY)
keypoints2, desc2 = sift.detectAndCompute(G2, None); # opencv 3

fnames = glob.glob('obj?.jpg')
fnames.sort()
for fname in fnames:

    I1 = cv2.imread(fname)
    G1 = cv2.cvtColor(I1,cv2.COLOR_BGR2GRAY)
    keypoints1, desc1 = sift.detectAndCompute(G1, None); # opencv 3
    height, width = I1.shape[:2]
    
    # brute-force matching
    bf = cv2.BFMatcher()

    # for each descriptor in desc1 find its
    # two nearest neighbours in desc2
    matches = bf.knnMatch(desc1,desc2, k=2)

    # distance ratio test
    alpha = 0.75
    good_matches = [m1 for m1,m2 in matches if m1.distance < alpha *m2.distance]
    
    points1 = [keypoints1[m.queryIdx].pt for m in good_matches]
    points1 = np.array(points1,dtype=np.float32)

    points2 = [keypoints2[m.trainIdx].pt for m in good_matches]
    points2 = np.array(points2,dtype=np.float32)
    
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)  # 5 pixels margin
    good_matches = [m for m, msk in zip(good_matches, mask) if msk == 1]

    pts = np.float32([ [0,0],
                       [0,height],
                       [width,height],
                       [width,0] ]).reshape(-1,1,2) # this has to be changed

    
    
    dst = cv2.perspectiveTransform(pts,H).reshape(4,2)
   
    J = I2.copy()
    cv2.line(J, (dst[0,0], dst[0,1]), (dst[1,0], dst[1,1]), (255,0,0),3)
    cv2.line(J, (dst[1,0], dst[1,1]), (dst[2,0], dst[2,1]), (255,0,0),3)
    cv2.line(J, (dst[2,0], dst[2,1]), (dst[3,0], dst[3,1]), (255,0,0),3)
    cv2.line(J, (dst[3,0], dst[3,1]), (dst[0,0], dst[0,1]), (255,0,0),3)


    I = cv2.drawMatches(I1,keypoints1,J,keypoints2,good_matches, None)

    cv2.imshow('keypoints',I)

    if cv2.waitKey() & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```
